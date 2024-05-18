import streamlit as st
from openai import OpenAI
import psycopg2
import os
from decimal import Decimal
from datetime import date
import dotenv

dotenv.load_dotenv()

# Set up OpenAI API credentials
client = OpenAI()

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


def query_to_vector(text, model="text-embedding-3-large"):
    """
    Converts text to a vector using OpenAI's embedding model.
    """
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding  # response["data"]["embedding"]


def query_postgresql(question):
    """
    Queries the PostgreSQL database with a vectorized form of the question.
    """
    vector = query_to_vector(question)
    vector_str = ",".join(
        map(str, vector)
    )  # Convert vector to a comma-separated string

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
    """
    Uses the GPT-4o model on Azure OpenAI endpoint to generate a conversational response.
    """
    messages = [
        {
            "role": "system",
            "content": "You are LobbyIQ's AI database assistant. Based on the user's question and relevant information retrieved, provide a detailed response. If there was not relevant information retrieved or you cannot answer confidently, let the user know.",
        },
        {"role": "user", "content": f"The provided context is: {context}"},
        {"role": "user", "content": f"The user's question is: {question}"},
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.3,
        messages=messages,
    )

    return response.choices[0].message.content.strip()


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
    page_title="LobbyIQ Registration Copilot",
    page_icon="https://ca.lobbyiq.org/static/media/LIQ_badge.2ead782603061026e6ed285580b71966.svg",
    layout="wide",
    initial_sidebar_state="auto",
)

# Set main logo on page
st.image(
    "https://ca.lobbyiq.org/static/media/LIQ_badge.2ead782603061026e6ed285580b71966.svg",
    width=56,
)

# Streamlit UI
st.title("LobbyIQ Registration Copilot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Enter your query (or type 'exit' to quit):"):
    if prompt.lower() == "exit":
        st.stop()

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        results = query_postgresql(prompt)
        context = "\n".join([str(result) for result in results])
        ai_response = gpt_4o_analysis(prompt, context)
        log_interaction(prompt, results, ai_response)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(f"**AI Response:**\n\n{ai_response}")
            st.markdown("### Retrieved Information")
            for result in results:
                if result["matching_reg_id_enr"]:
                    result["matching_reg_id_enr"] = (
                        str(result["matching_reg_id_enr"])
                        .replace("Decimal('", "")
                        .replace("')", "")
                        .replace("[", "")
                        .replace("]", "")
                    )

                st.markdown(f"#### **Client:** {result['client_clean_coalesce']}")
                st.markdown(f"**Firm:** {result['firm_clean_coalesce']}")
                st.markdown(
                    f"**Matching Registration ID:** {result['matching_reg_id_enr']}"
                )

                url = f"https://lobbycanada.gc.ca/app/secure/ocl/lrs/do/rgstrnGvrnmntInstttns?regId={result['matching_reg_id_enr']}"
                # Print URL as a hyperlink
                st.markdown(f"**URL:** {url}")

                st.markdown(f"**Description:** {result['description']}")
                # st.markdown(
                #     f"**Effective Date:** {result['matching_effective_date_vigueur']}"
                # )
                # st.markdown(f"**End Date:** {result['matching_end_date_fin']}")
                st.markdown(
                    f"**Confidence / Vector Similarity Score:** {100*result['score']:.2f}%"
                )
                st.markdown(
                    "\n*---------------------------------------------------------------------------*\n"
                )

            raw_registration_ids = [
                str(result["matching_reg_id_enr"]) for result in results
            ]
            st.markdown("\n\n**All Relevant Registration IDs:**")
            st.markdown(", ".join(raw_registration_ids))

            st.markdown("\n\n**All Relevant Registration IDs in URL pattern:**")
            for registration_id in raw_registration_ids:
                url = f"https://lobbycanada.gc.ca/app/secure/ocl/lrs/do/rgstrnGvrnmntInstttns?regId={registration_id}"
                st.markdown(f"[Registration {registration_id}]({url})")

            st.markdown("\n\n\n\n")  # Add an empty line for readability
            st.markdown("\n\n\n\n\n\n\n\n\n\n\n")  # Add an empty line for readability

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": ai_response})

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.session_state.messages.append(
            {"role": "assistant", "content": f"An error occurred: {e}"}
        )



