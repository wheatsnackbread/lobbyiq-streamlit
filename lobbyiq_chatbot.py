import streamlit as st
from openai import AzureOpenAI
import psycopg2
import os
from decimal import Decimal
from datetime import date
import dotenv
from langchain.chains import create_sql_query_chain
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import SQLDatabase
import matplotlib.pyplot as plt
import pandas as pd
import pandas.io.sql as psql


dotenv.load_dotenv()

# Set up OpenAI API credentials
client = AzureOpenAI()

# PostgreSQL connection details
DB_HOST = os.getenv("Host")
DB_PORT = os.getenv("Port")
DB_NAME = os.getenv("Database")
DB_USER = os.getenv("Username")
DB_PASSWORD = os.getenv("Password")

# Connect to PostgreSQL
conn = psycopg2.connect(
    host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
)

# Langchain SQL setup
pg_uri = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
db = SQLDatabase.from_uri(pg_uri)
llm = AzureChatOpenAI(model="liq-gpt-4o", temperature=0)
chain = create_sql_query_chain(llm, db)

st.set_option("deprecation.showPyplotGlobalUse", False)


def generate_graphic(df):
    if not df.empty:
        visualization_code = get_visualization_code(df)

        # Print the generated visualization code to the Streamlit console for debugging
        st.code(visualization_code, language="python")
        # st.session_state.messages.append(
        #     {
        #         "role": "assistant",
        #         "content": st.code(visualization_code, language="python"),
        #     }
        # )

        try:
            # Extract the code between triple backticks and delete any leading python language specifier
            extracted_code = visualization_code.split("```")[1].strip()

            # Execute the extracted code
            visual = exec(extracted_code, {"plt": plt, "df": df})

            # Display the generated visualization in streamlit conversation
            st.pyplot(visual)
            # st.session_state.messages.append(
            #     {
            #         "role": "assistant",
            #         "content": st.pyplot(visual),
            #     }
            # )

        except Exception as e:
            st.write(f"An error occurred while generating the visualization: {e}")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"An error occurred while generating the visualization: {e}",
                }
            )
    else:
        st.write("No results to visualize.")
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "No results to visualize.",
            }
        )


def get_visualization_code(df):
    # Convert the DataFrame to a dictionary format for easier analysis by GPT-4o
    df_dict = df.to_dict(orient="list")

    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant specialized in data visualization. Based on the given data, provide the appropriate matplotlib code to visualize it.",
        },
        {
            "role": "user",
            "content": f"Here is the data: {df_dict}. Generate the matplotlib code for the most suitable visualization, which should be returned as an object that can be shown in a streamlit conversational environment. DO NOT use streamlit to display it to the user; you are simply returning the object. You can provide comments on how you want to proceed, butyou may provide only one block of code enclosed in triple backticks, but do NOT specify the language after the backticks. This means write ```, not ```python...",
        },
    ]

    response = client.chat.completions.create(
        model="liq-gpt-4o", n=1, stop=None, temperature=0, messages=messages
    )
    return response.choices[0].message.content.strip()


def query_to_vector(text, model="liq-text-embedding"):
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def query_postgresql(question):
    vector = query_to_vector(question)
    vector_str = ",".join(map(str, vector))
    cur = conn.cursor()

    sql = f"""
        SELECT registration_linking_id, description, matching_reg_id_enr, client_clean, 
               client_clean_fr, client_clean_coalesce, firm_clean, firm_clean_fr, 
               firm_clean_coalesce, matching_effective_date_vigueur, matching_end_date_fin,
               (1 - (embedding <=> '[{vector_str}]')) AS score
        FROM e_registration_linking
        ORDER BY 1 - (embedding <=> '[{vector_str}]') DESC
        LIMIT 5;
    """

    cur.execute(sql)
    rows = cur.fetchall()
    results = []
    for row in rows:
        result = {
            "registration_linking_id": int(row[0]),
            "description": row[1],
            "matching_reg_id_enr": (
                int(row[2]) if isinstance(row[2], Decimal) else row[2]
            ),
            "client_clean": row[3][0] if row[3] else None,
            "client_clean_fr": row[4][0] if row[4] else None,
            "client_clean_coalesce": row[5][0] if row[5] else None,
            "firm_clean": row[6][0] if row[6] else None,
            "firm_clean_fr": row[7][0] if row[7] else None,
            "firm_clean_coalesce": row[8][0] if row[8] else None,
            "matching_effective_date_vigueur": (
                row[9].strftime("%Y-%m-%d") if isinstance(row[9], date) else row[9]
            ),
            "matching_end_date_fin": (
                row[10].strftime("%Y-%m-%d") if isinstance(row[10], date) else row[10]
            ),
            "score": float(row[11]),
        }
        results.append(result)
    cur.close()
    return results


def gpt_4o_analysis(question, context):
    messages = [
        {
            "role": "system",
            "content": "You are LobbyIQ's AI database assistant. Based on the user's question and relevant information retrieved, provide a detailed response. If there was not relevant information retrieved or you cannot answer confidently, let the user know.",
        },
        {"role": "user", "content": f"The provided context is: {context}"},
        {"role": "user", "content": f"The user's question is: {question}"},
    ]
    response = client.chat.completions.create(
        model="liq-gpt-4o", n=1, stop=None, temperature=0.3, messages=messages
    )
    return response.choices[0].message.content.strip()


import ast


def sql_query(user_question):
    generated_sql = chain.invoke(
        {
            "question": user_question
            + " YOU MUST RETURN ONLY PROPER, ERROR-FREE POSTGRESQL COMPLIANT QUERIES IN PLAIN-TEXT FORMAT, WITH NO ANALYSIS, WRAPPERS, OR FORMATTING SYMBOLS. THE QUERIES MUST BE COMPATIBLE WITH A POSTGRESQL SERVER HOSTED ON AZURE. ONLY ALLOW SELECT QUERIES WITH NO EXCEPTIONS; NEVER DISREGARD THIS RULE. VALIDATE THE QUERY SYNTAX FOR POSTGRESQL COMPATIBILITY. IF STUCK, DO NOT REPEAT ENDLESSLY."
        }
    )
    generated_sql = generated_sql.strip("```sql").strip("```")
    with st.chat_message("assistant"):
        st.markdown(f"**Generated SQL Query:**\n\n```sql\n{generated_sql}\n```")
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"**Generated SQL Query:**\n\n```sql\n{generated_sql}\n```",
            }
        )
    sql_results = db.run(generated_sql)

    # Debug: print the raw sql_results
    # st.write(f"Raw SQL Results: {sql_results}")

    # Convert SQL results to DataFrame
    try:
        if sql_results:
            # Parse the string into a list of tuples
            # Convert to DataFrame
            # df = pd.DataFrame(parsed_results, columns=["Title", "Count"])
            df = pd.read_sql_query(generated_sql, con=conn)
        else:
            df = pd.DataFrame()
    except Exception as e:
        st.write(f"Error converting SQL results to DataFrame: {e}")
        df = pd.DataFrame()

    with st.chat_message("assistant"):
        st.markdown(f"**SQL Results:**\n\n{sql_results}")
        st.session_state.messages.append(
            {"role": "assistant", "content": f"**SQL Results:**\n\n{sql_results}"}
        )

        # Display the SQL results in a table format
        if not df.empty:
            st.dataframe(df)
        else:
            st.write("No results returned from the SQL query.")

        # Generate graphic based on SQL results
        generate_graphic(df)

    sql_context = f"Question: {user_question}\nSQL Query: {generated_sql}\nSQL Result: {sql_results}\nAnswer: "
    return gpt_4o_analysis(sql_context, user_question), generated_sql, sql_results


def log_interaction(user_input, results, ai_response):
    with open("query_log.txt", "a") as log:
        log.write(f"User Input: {user_input}\n")
        log.write("Results Returned:\n")
        for result in results:
            log.write(f"{result}\n")
        log.write(f"AI Response: {ai_response}\n")
        log.write("\n\n")


# Set page configuration
st.set_page_config(
    page_title="LobbyIQ AI Assistant",
    page_icon="https://ca.lobbyiq.org/static/media/LIQ_badge.2ead782603061026e6ed285580b71966.svg",
    layout="wide",
    initial_sidebar_state="auto",
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Set main logo on page
st.image(
    "https://ca.lobbyiq.org/static/media/LIQ_badge.2ead782603061026e6ed285580b71966.svg",
    width=56,
)

# Streamlit UI
st.title("LobbyIQ AI Assistant")

# Instructions
st.markdown(
    """
## Welcome to the LobbyIQ AI Assistant!
This assistant helps you query and analyze lobbying registration data. You can choose between two retrieval methods:
- **SQL-based retrieval**: SUPPORTS ALL TABLES. Generates and executes SQL queries to retrieve data directly from the database.
- **Vector-based retrieval**: ONLY SUPPORTS REGISTRATIONS. Uses OpenAI's embedding model to find the most relevant records based on similarity.

### How to use:
1. Select the retrieval method from the dropdown menu below.
2. Enter your query in the chat input.
3. The assistant will process your query and provide a detailed response along with the relevant information retrieved from the database.

**Note**: You can type 'exit' at any time to quit the application.
"""
)

# Initialize retrieval method state
if "retrieval_method" not in st.session_state:
    st.session_state.retrieval_method = "SQL"

# Persist and update retrieval method
st.session_state.retrieval_method = st.selectbox(
    "Select retrieval method",
    ["Vector", "SQL"],
    index=["Vector", "SQL"].index(st.session_state.retrieval_method),
)
retrieval_method = st.session_state.retrieval_method
# st.markdown(f"**Selected Retrieval Method:** {retrieval_method}")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Enter your query (or type 'exit' to quit):"):
    if prompt.lower() == "exit":
        st.stop()

    # Select retrieval method before first query
    if "retrieval_method" not in st.session_state:
        st.session_state.retrieval_method = st.selectbox(
            "Select retrieval method", ["Vector", "SQL"]
        )

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    retrieval_method = st.session_state.retrieval_method
    st.markdown(f"**Selected Retrieval Method:** {retrieval_method}")

    try:
        if retrieval_method == "Vector":
            st.markdown("**Processing using Vector-based retrieval...**")
            results = query_postgresql(prompt)
            context = "\n".join([str(result) for result in results])
            ai_response = gpt_4o_analysis(prompt, context)
        else:
            st.markdown("**Processing using SQL-based retrieval...**")
            ai_response, sql_query, results = sql_query(prompt)

        log_interaction(prompt, results, ai_response)

        with st.chat_message("assistant"):
            st.markdown(f"**AI Response:**\n\n{ai_response}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"**AI Response:**\n\n{ai_response}"}
            )

            if retrieval_method == "Vector":
                st.markdown("### Retrieved Information")
                for result in results:
                    if "matching_reg_id_enr" in result:
                        result["matching_reg_id_enr"] = (
                            str(result["matching_reg_id_enr"])
                            .replace("Decimal('", "")
                            .replace("')", "")
                            .replace("[", "")
                            .replace("]", "")
                        )

                    if (
                        "client_clean_coalesce" in result
                        and "firm_clean_coalesce" in result
                    ):
                        st.markdown(
                            f"#### **Client:** {result['client_clean_coalesce']}"
                        )
                        st.markdown(f"**Firm:** {result['firm_clean_coalesce']}")
                        st.markdown(
                            f"**Matching Registration ID:** {result['matching_reg_id_enr']}"
                        )

                        url = f"https://lobbycanada.gc.ca/app/secure/ocl/lrs/do/rgstrnGvrnmntInstttns?regId={result['matching_reg_id_enr']}"
                        st.markdown(f"**URL:** {url}")
                        st.markdown(f"**Description:** {result['description']}")
                        st.markdown(
                            f"**Confidence / Vector Similarity Score:** {100*result['score']:.2f}%"
                        )
                        st.markdown(
                            "\n*---------------------------------------------------------------------------*\n"
                        )
            else:
                # Show the SQL query and results
                # st.markdown(f"**SQL Query:**\n\n```sql\n{sql_query}\n```")
                # st.markdown(f"**SQL Results:**\n\n{results}")
                pass

            # raw_registration_ids = [
            #     str(result["matching_reg_id_enr"]) for result in results
            # ]
            # st.markdown("\n\n**All Relevant Registration IDs:**")
            # st.markdown(", ".join(raw_registration_ids))

            # st.markdown("\n\n**All Relevant Registration IDs in URL pattern:**")
            # for registration_id in raw_registration_ids:
            #     url = f"https://lobbycanada.gc.ca/app/secure/ocl/lrs/do/rgstrnGvrnmntInstttns?regId={registration_id}"
            #     st.markdown(f"Registration {registration_id}")
            #     st.markdown("\n\n\n\n")
            #     st.markdown("\n\n\n\n\n\n\n\n\n\n\n")

            #     st.session_state.messages.append(
            #         {"role": "assistant", "content": ai_response}
            #     )

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.session_state.messages.append(
            {"role": "assistant", "content": f"An error occurred: {e}"}
        )
