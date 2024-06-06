def follow_up(query):
    user_question = query
    import streamlit as st

    # from openai import OpenAI
    from openai import AzureOpenAI
    import psycopg2
    import os
    from decimal import Decimal
    from datetime import date
    import dotenv

    dotenv.load_dotenv()

    # Set up OpenAI API credentials
    # client = OpenAI()
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
            model="liq-gpt-4o",  # for Azure
            # model="gpt-4o", # for OpenAI
            # max_tokens=500,
            n=1,
            stop=None,
            temperature=0,
            messages=messages,
        )

        return response.choices[0].message.content.strip()

    from langchain.chains import create_sql_query_chain
    from langchain_openai import AzureChatOpenAI

    pg_uri = (
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    from langchain_community.utilities import SQLDatabase

    db = SQLDatabase.from_uri(pg_uri)
    print("The detected DB dialect is " + str(db.dialect))
    print("The detected DB tables are " + str(db.get_usable_table_names()))

    # liq-gpt-4o; liq-gpt-35
    llm = AzureChatOpenAI(model="liq-gpt-4o", temperature=0)

    chain = create_sql_query_chain(llm, db)
    generated_sql = chain.invoke(
        {
            "question": user_question
            + " YOU MUST RETURN ONLY PROPER, ERROR-FREE POSTGRESQL COMPLAINT QUERIES IN PLAIN-TEXT FORMAT, WITH NO ANALYSIS, WRAPPERS, OR FORMATTING SYMBOLS."
        }
    )
    print(
        "\n\n\n\nHere is the generated query to be executed against the database. "
        + generated_sql
    )

    # Run the query
    # keep only the content in generated_sql between ```sql ```
    generated_sql = generated_sql.strip("```sql").strip("```")
    sql_results = db.run(generated_sql)
    print("The results of the SQL query are " + str(sql_results))

    sql_context = f"""Given the following user question, corresponding SQL query, and SQL result, answer the user question.

    Question: {user_question}
    SQL Query: {generated_sql}
    SQL Result: {sql_results}
    Answer: """

    # usage: gpt_4o_analysis(context, question)
    final_answer = gpt_4o_analysis(sql_context, user_question)
    print(final_answer)

    return final_answer
