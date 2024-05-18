from openai import OpenAI
import psycopg2
import os
from decimal import Decimal
from datetime import date
import dotenv

dotenv.load_dotenv()

# # Set up OpenAI API credentials
# openai.api_key = os.getenv("OPENAI_KEY")

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
    prompt = f"""
    You are an AI assistant. The user has asked the following question:
    "{question}"
    
    Based on the information retrieved, provide a detailed and conversational response.
    
    Information:
    {context}
    
    Response:
    """

    # Rewrite prompt as messages in this format:
    # messages=[
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "Hello!"}
    # ]

    messages = [
        {
            "role": "system",
            "content": "You are LobbyIQ's AI database assistant. Based on the user's question and relevant information retrieved, provide a detailed response. If there was not relevant information retrieved or you cannot answer confidently, let the user know.",
        },
        # now, you need to add in the context as the user, indicating which is the context and which is the question
        {"role": "user", "content": "\n\nThe provided context is: " + context},
        {"role": "user", "content": "\n\nThe user's question is :" + question},
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


# Log file setup
LOG_FILE = "query_log.txt"


def log_interaction(user_input, results, ai_response):
    with open(LOG_FILE, "a") as log:
        log.write(f"User Input: {user_input}\n")
        log.write("Results Returned:\n")
        for result in results:
            log.write(f"{result}\n")
        log.write(f"AI Response: {ai_response}\n")
        log.write("\n\n")


# CLI Loop
while True:
    user_input = input("Enter your query (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    try:
        results = query_postgresql(user_input)
        context = "\n".join([str(result) for result in results])
        ai_response = gpt_4o_analysis(user_input, context)
        log_interaction(user_input, results, ai_response)
        print(ai_response)
    except Exception as e:
        print(f"An error occurred: {e}\n\n\n")


# demo text
#  Can you help me find some registrations related to the dairy and milk industry? Who are the main players, and can we develop three key pillars that they all care about for our lobbying platform?
