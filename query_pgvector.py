import openai
import psycopg2
import os
from decimal import Decimal
from datetime import date

# Set up OpenAI API credentials
openai.api_key = os.getenv("OPENAI_KEY")

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
    response = openai.Embedding.create(input=[text], model=model)
    return response["data"][0]["embedding"]


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


# CLI Loop
while True:
    user_input = input("Enter your query (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    try:
        results = query_postgresql(user_input)
        print("\n\n***RESULTS RETURNED***\n\nTop matching registration IDs:\n")
        for result in results:

            # cleaned matching registration ID
            if result["matching_reg_id_enr"]:
                # change the format [Decimal('949534')] to '949534'
                result["matching_reg_id_enr"] = (
                    str(result["matching_reg_id_enr"])
                    .replace("Decimal('", "")
                    .replace("')", "")
                    .replace("[", "")
                    .replace("]", "")
                )

            # print(f"Registration Linking ID: {result['registration_linking_id']}\n")
            print(f"Matching Registration ID: {result['matching_reg_id_enr']}\n")
            # print(f"Client Clean: {result['client_clean']}\n")
            # print(f"Client Clean (FR): {result['client_clean_fr']}\n")
            print(f"Client: {result['client_clean_coalesce']}\n")
            # print(f"Firm Clean: {result['firm_clean']}\n")
            # print(f"Firm Clean (FR): {result['firm_clean_fr']}\n")
            print(f"Firm: {result['firm_clean_coalesce']}\n")

            print(f"Description: {result['description']}\n")

            print(f"Effective Date: {result['matching_effective_date_vigueur']}\n")
            print(f"End Date: {result['matching_end_date_fin']}\n")
            print(f"Confidence / Vector Similarity Score: {result['score']:.6f}\n")
            print(
                "\n*---------------------------------------------------------------------------*\n"
            )

        raw_registration_ids = [
            str(result["matching_reg_id_enr"]) for result in results
        ]
        print("\n\nAll Relevant Registration IDs:")
        print(raw_registration_ids)

        print("\n\nAll Relevant Registration IDs in URL pattern:")
        for registration_id in raw_registration_ids:
            url = f"https://lobbycanada.gc.ca/app/secure/ocl/lrs/do/rgstrnGvrnmntInstttns?regId={registration_id}"
            print(url)
        print("\n\n\n\n")  # Add an empty line for readability
        print("\n\n\n\n\n\n\n\n\n\n\n")  # Add an empty line for readability

    except Exception as e:
        print(f"An error occurred: {e}\n\n\n")

# Can you help me find registrations that are interested in the CCA rate interpretations related to newly-constructured facilities in the natural resource extraction industry?
# Can you find me some registrations which are related to legislation in the province of Ontario related to dairy farming and regulation, pertaining to milk in particular?
