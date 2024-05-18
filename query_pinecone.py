import openai
from pinecone import Pinecone
import os

# Set up OpenAI and Pinecone API credentials
openai.api_key = os.getenv("OPENAI_KEY")
PINECONE_KEY = os.getenv("PINECONE_KEY")

# Connect to an existing Pinecone index
index_name = "liq-registrations"
pc = Pinecone(api_key=PINECONE_KEY)

index = pc.Index(name=index_name)


def query_to_vector(text, model="text-embedding-3-large"):
    """
    Converts text to a vector using OpenAI's embedding model.
    """
    response = openai.Embedding.create(input=[text], model=model)
    return response["data"][0]["embedding"]


def query_pinecone(question, metadata_filter, filter_field, filter_variations):
    """
    Queries the Pinecone index with a vectorized form of the question.
    """
    vector = query_to_vector(question)
    response = ""
    if metadata_filter:
        # filter query should check if the filter field contains any of the filter variations
        # here is an example of what the filter query argument looks like
        #   "genre": { "$in": ["comedy", "documentary", "drama"] }
        filter_query = {filter_field: {"$in": filter_variations}}

        response = index.query(
            vector=vector, top_k=5, include_metadata=True, filter=filter_query
        )
    else:
        response = index.query(vector=vector, top_k=5, include_metadata=True)
    results = []
    for match in response["matches"]:
        result = {
            "registration_id": match["metadata"]["matching_reg_id_enr"],
            "description": match["metadata"]["description"],
            "matching_reg_id_enr": match["metadata"]["matching_reg_id_enr"],
            "matching_reg_type_enr": match["metadata"]["matching_reg_type_enr"],
            "matching_reg_num_enr": match["metadata"]["matching_reg_num_enr"],
            "matching_en_client_org_corp_nm_an": match["metadata"][
                "matching_en_client_org_corp_nm_an"
            ],
            "matching_fr_client_org_corp_nm": match["metadata"][
                "matching_fr_client_org_corp_nm"
            ],
            "matching_effective_date_vigueur": match["metadata"][
                "matching_effective_date_vigueur"
            ],
            "matching_end_date_fin": match["metadata"]["matching_end_date_fin"],
            "matching_rgstrnt_last_nm_dclrnt": match["metadata"][
                "matching_rgstrnt_last_nm_dclrnt"
            ],
            "matching_rgstrnt_1st_nm_prenom_dclrnt": match["metadata"][
                "matching_rgstrnt_1st_nm_prenom_dclrnt"
            ],
            "score": match["score"],
        }
        results.append(result)
    return results


# CLI Loop
while True:
    user_input = input(
        "Enter your query (or type 'exit' to quit, 'filter' to apply metadata filters): "
    )
    if user_input.lower() == "exit":
        break
    try:
        metadata_filter = False
        filter_input = ""
        filter_field = ""
        filter_variations = list()
        if user_input.lower() == "filter":
            metadata_filter = True
            filter_input = input(
                "\nWhich field do you want to filter on? Enter the number. \n 1. Registrant's last name \n 2. Registrant's first name\n --> "
            )
            filter_field = ""
            if filter_input == "1":
                filter_field = "matching_rgstrnt_last_nm_dclrnt"
            elif filter_input == "2":
                filter_field = "matching_rgstrnt_1st_nm_prenom_dclrnt"
            else:
                print("Invalid input. Exiting.")
                break
            filter_value = input(
                "What value do you want to filter for? Input in lowercase with no spaces: "
            )

            # The filter_value will be something like rose. We now create a list called filter_variations which contains every possible combination of the letters in either uppercase or lowercase mix. That includes rose, Rose, ROse, ROSe, etc.
            filter_variations = []
            for i in range(2 ** len(filter_value)):
                filter_variations.append(
                    "{"
                    + "".join(
                        [
                            filter_value[j].upper() if (i >> j) & 1 else filter_value[j]
                            for j in range(len(filter_value))
                        ]
                    )
                    + "}"
                )
            print("Here are the filter varitiations: " + str(filter_variations))
            user_input = input("\nEnter your general query about registrations: ")

        results = query_pinecone(
            user_input, metadata_filter, filter_field, filter_variations
        )
        print("\n\n***RESULTS RETURNED***\n\nTop matching registration IDs:\n")
        for result in results:
            print(f"Registration ID: {result['registration_id']}\n")

            # Print the remaining sections as well: matching_reg_id_enr,matching_reg_type_enr,matching_reg_num_enr,matching_en_client_org_corp_nm_an,matching_fr_client_org_corp_nm,matching_effective_date_vigueur,matching_end_date_fin,matching_rgstrnt_last_nm_dclrnt,matching_rgstrnt_1st_nm_prenom_dclrnt

            print(f"English firm name: {result['matching_en_client_org_corp_nm_an']}\n")

            print(
                f"Individual last names: {result['matching_rgstrnt_last_nm_dclrnt']}\n"
            )
            print(
                f"Individual first names: {result['matching_rgstrnt_1st_nm_prenom_dclrnt']}\n"
            )

            print(f"Description: {result['description']}\n")
            print(f"Confidence / vector similarity score: {result['score']}\n")

            associated_registration_ids = (
                result["registration_id"]
                .replace("{", "")
                .replace("}", "")
                .replace(".0", "")
                .split(",")
            )
            for registration_id in associated_registration_ids:
                url = f"https://lobbycanada.gc.ca/app/secure/ocl/lrs/do/rgstrnGvrnmntInstttns?regId={registration_id}"
                print(f"URL: {url}")

            print(
                "\n*---------------------------------------------------------------------------*\n"
            )

        raw_registration_ids = [result["registration_id"] for result in results]
        # This will return something like ['{843976.0}', '{841100.0,822718.0,823010.0,823067.0}', '{822717.0,863972.0,860275.0}', '{835913.0}', '{870425.0}']
        # We will clean this up so that each registration ID is a separate string without decimal points or curly braces. This includes separating out grouped registration IDs.
        registration_ids = []
        for raw_registration_id in raw_registration_ids:
            cleaned_ids = (
                raw_registration_id.replace("{", "")
                .replace("}", "")
                .replace(".0", "")
                .split(",")
            )
            registration_ids.extend(cleaned_ids)

        print("\n\nAll Relevant Registration IDs:")
        print(registration_ids)

        print("\n\nAll Relevant Registration IDs in URL pattern:")
        for registration_id in registration_ids:
            url = f"https://lobbycanada.gc.ca/app/secure/ocl/lrs/do/rgstrnGvrnmntInstttns?regId={registration_id}"
            print(url)
        print("\n\n\n\n")  # Add an empty line for readability
        print("\n\n\n\n\n\n\n\n\n\n\n")  # Add an empty line for readability

    except Exception as e:
        print(f"An error occurred: {e}\n\n\n")


### DEMO CONTENT

# 1244,arranging meetings on taxation and finance issues as they relates to questions regarding taxation of and capital cost allowance (cca) classification of new facilities. specifically requesting an interpretation on the cca rate that might be applied to a new natural gas and liquids processing plant.,"{841100.0,822718.0,823010.0,823067.0}"


# Can you help me find registrations that are interested in the CCA rate interpretations related to newly-constructured facilities in the natural resource extraction industry?
# Can you find me some registrations which are related to legislation in the province of Ontario related to dairy farming and regulation, pertaining to milk in particular?
