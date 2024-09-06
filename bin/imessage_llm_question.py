import vobject
import sqlite3
import pandas as pd
import re
from datetime import datetime
import argparse
from pandas import Timestamp
import google.generativeai as genai
import os
from rich.console import Console
from rich.markdown import Markdown
import tiktoken 


def create_detailed_llm_prompt(filtered_messages, question):
    # Construct the prompt
    prompt = f"Question: {question}\n\nMessage History:\n"
    for message in filtered_messages:
        sender = message.get('sender_name', 'Unknown Sender')
        sender_number = message.get('sender_number', 'Unknown Number')
        timestamp = message.get('timestamp', 'Unknown Time')
        text = message.get('text', 'No Text')
        attachment = message.get('attached_file', None)
        people_in_chat = ", ".join([f"{person[0]} ({person[1]})" for person in message.get('people_in_chat', [])])

        # Basic message information
        prompt += (
            f"From: {sender} ({sender_number}) at {timestamp}\n"
            f"People in Chat: {people_in_chat}\n"
            f"Message: {text}\n"
        )
        
        # Conditionally include attachment information
        if attachment:
            prompt += f"Attachment: {attachment}\n"
        
        # Add a blank line after each message
        prompt += "\n"
    
    # Token estimation
    encoding = tiktoken.get_encoding("cl100k_base")  # Use appropriate encoding
    num_tokens = len(encoding.encode(prompt, disallowed_special=()))
    
    return prompt, num_tokens


def standardize_phone_number(phone_number):
    phone_number = str(phone_number)
    
    if re.search('[a-zA-Z]', phone_number):
        return phone_number
    
    digits = re.sub(r'\D', '', phone_number)
    
    if len(digits) == 10:  # US number without country code
        return f"+1{digits}"
    elif len(digits) == 11 and digits.startswith('1'):  # US number with country code
        return f"+{digits}"
    elif len(digits) > 11:  # International number
        # Assume the first 1-3 digits are the country code
        for i in range(1, 4):
            potential_country_code = digits[:i]
            if potential_country_code in ['1', '7', '20', '27', '30', '31', '32', '33', '34', '36', '39', '40', '41', '43', '44', '45', '46', '47', '48', '49', '51', '52', '53', '54', '55', '56', '57', '58', '60', '61', '62', '63', '64', '65', '66', '81', '82', '84', '86', '90', '91', '92', '93', '94', '95', '98']:
                return f"+{digits}"
    
    # If we can't determine the format, return the original number
    return phone_number


def vcf_to_dataframe(vcf_path):
    names = []
    phone_numbers = []
    with open(vcf_path, 'r') as vcf_file:
        for vcard in vobject.readComponents(vcf_file.read()):
            # Extract the contact name
            try:
                name = vcard.fn.value
            except:
                print(vcard)
                continue
            # Initialize a variable for phone number for each contact
            phone_number = None
            if hasattr(vcard, 'tel'):
                phone_number = vcard.tel.value
                if isinstance(vcard.tel, list):
                    phone_number = vcard.tel[0].value
            names.append(name)
            phone_numbers.append(phone_number)
    contacts_df = pd.DataFrame({
        'contact_name': names,
        'phone_number': phone_numbers
    })
    return contacts_df


def conversion_function(attributed_body: bytes) -> str:
    """
    Convert an attributedBody field from the iMessage database to a plain text string.
    """
    # Decode the bytes using ISO-8859-1 to preserve all byte data
    decoded_string = attributed_body.decode('ISO-8859-1')
    # Locate the position where the actual text starts and ends based on observed byte patterns
    start_marker = "\x01+"  # Observed start pattern for text
    end_marker = "\x86"     # Observed end pattern for text
    start_index = decoded_string.find(start_marker)
    end_index = decoded_string.find(end_marker, start_index)
    if start_index != -1 and end_index != -1:
        # Extract the text by slicing between start and end markers
        start_index += len(start_marker)
        text = decoded_string[start_index:end_index]
        # Strip any leading non-printable or unexpected characters
        text = re.sub(r'^[^\x20-\x7E]+', '', text)  # Strips all non-ASCII characters
        # Replace encoded special characters with their plain text equivalents
        text = text.replace('\xe2\x80\x9c', '“')
        text = text.replace('\xe2\x80\x9d', '”')
        text = text.replace('\xe2\x80\x99', '’')
        return text.strip()
    return ""



def get_llm_prompt(question, start_date, end_date):
    filtered_messages = get_filtered_imessages(start_date, end_date)
    prompt, token_count = create_detailed_llm_prompt(filtered_messages, question)
    return prompt, token_count, filtered_messages


def get_filtered_imessages(start_date, end_date):
    home_directory = os.path.expanduser("~")
    db_path = os.path.join(home_directory, "Downloads", "imessage_data.db")
    vcf_path = os.path.join(home_directory, "Downloads", "all_contacts_9_6_24.vcf")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(" select name from sqlite_master where type = 'table' ")
    #messages = pd.read_sql_query('''select *, datetime(date/1000000000 + strftime("%s", "2001-01-01") ,"unixepoch","localtime")  as date_utc from message''', conn) 
    messages = pd.read_sql_query('''
        SELECT 
            *, 
            datetime(date/1000000000 + strftime("%s", "2001-01-01") ,"unixepoch","localtime") as date_utc,
            attributedBody
        FROM message
    ''', conn)

    handles = pd.read_sql_query("select * from handle", conn)
    chat_message_joins = pd.read_sql_query("select * from chat_message_join", conn)
    chat_handle_joins = pd.read_sql_query("SELECT * FROM chat_handle_join", conn)
    messages['message_date'] = messages['date']
    messages['timestamp'] = messages['date_utc'].apply(lambda x: pd.Timestamp(x))
    messages['date'] = messages['timestamp'].apply(lambda x: x.date())
    messages['month'] = messages['timestamp'].apply(lambda x: int(x.month))
    messages['year'] = messages['timestamp'].apply(lambda x: int(x.year))
    messages.rename(columns={'ROWID' : 'message_id'}, inplace = True)
    handles.rename(columns={'id' : 'phone_number', 'ROWID': 'handle_id'}, inplace = True)


    df_messages = pd.merge(messages, handles, on='handle_id', how='left')
    df_messages = pd.merge(df_messages, chat_message_joins, on='message_id', how='left')
    # Merge chat_handle_joins to get all participants in the chat
    df_messages = pd.merge(df_messages, chat_handle_joins, on='chat_id', how='left', suffixes=('', '_recipient'))
    # Filter out recipient handle_id that matches sender handle_id for one-on-one chats

    df_messages['text'] = df_messages['text'].astype(str)
    df_messages_filtered = df_messages[df_messages['text'].str.strip().astype(bool)]
    df_messages_filtered = df_messages_filtered.reset_index(drop=True)

    #all_contacts.vcf
    contacts_df = vcf_to_dataframe(vcf_path)


    # Standardize phone numbers in advance for both dataframes
    contacts_df['standardized_phone_number'] = contacts_df['phone_number'].apply(standardize_phone_number)
    df_messages_filtered['standardized_phone_number_of_sender'] = df_messages_filtered["phone_number"].apply(standardize_phone_number)

    df_messages_filtered = df_messages_filtered.merge(
        contacts_df[['standardized_phone_number', 'contact_name']],
        left_on='standardized_phone_number_of_sender',
        right_on='standardized_phone_number',
        how='left'
    ).rename(columns={'contact_name': 'sender_contact_name'})

    debug_phone_number = '+16505345255'
    debug_row = df_messages_filtered[df_messages_filtered['standardized_phone_number_of_sender'] == debug_phone_number]

    # Drop the now redundant `standardized_phone_number` column from `contacts_df` in the merged dataframe
    df_messages_filtered = df_messages_filtered.drop(columns=['standardized_phone_number'])

    attachments = pd.read_sql_query("SELECT * FROM attachment", conn)
    message_attachment_joins = pd.read_sql_query("SELECT * FROM message_attachment_join", conn)

    # Merge to get full details of attachments linked to each message
    df_messages_filtered = pd.merge(df_messages_filtered, message_attachment_joins, on='message_id', how='left')
    df_messages_filtered = pd.merge(df_messages_filtered, attachments, left_on='attachment_id', right_on='ROWID', how='left')

    chat_handle_joins = chat_handle_joins.merge(
        handles[['handle_id', 'phone_number']],
        on='handle_id',
        how='left'
    )
    chat_handle_joins['standardized_phone_number'] = chat_handle_joins['phone_number'].apply(standardize_phone_number)

    # Merge with contacts_df to get the contact names
    chat_handle_joins = chat_handle_joins.merge(
        contacts_df[['standardized_phone_number', 'contact_name']],
        on='standardized_phone_number',
        how='left'
    )
    chat_handle_joins['contact_name'] = chat_handle_joins['contact_name'].fillna('Unknown')

    # Group by chat_id and aggregate contact details
    participant_details_df = chat_handle_joins.groupby('chat_id').agg({
        'contact_name': list,
        'standardized_phone_number': list
    }).reset_index()

    # Merge the messages with participant details
    df_messages_filtered = df_messages_filtered.merge(
        participant_details_df,
        on='chat_id',
        how='left'
    )

    def calculate_token_count(text):
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text, disallowed_special=()))


    df_messages_filtered['token_count'] = df_messages_filtered['text'].apply(calculate_token_count)

    # Combine text and participant details into a single list
    df_messages_filtered['message_and_participants'] = df_messages_filtered.apply(
        lambda row: {
            'text': row['text'] if row['text'] != "None" else (conversion_function(row["attributedBody"].iloc[0]) if isinstance(row["attributedBody"].iloc[0], bytes) else None),
            'timestamp': row['timestamp'],
            'sender_name': row['sender_contact_name'] if not row['is_from_me'] else "Dustin Axman",
            'sender_number': row['standardized_phone_number_of_sender'] if not row['is_from_me'] else '+15855766294',
            'attached_file': None if pd.isna(row['filename']) else row['filename'],
            'people_in_chat': tuple(sorted(list(zip(
                row['contact_name'] if isinstance(row['contact_name'], list) else [row['contact_name']],
                row['standardized_phone_number'] if isinstance(row['standardized_phone_number'], list) else [row['standardized_phone_number']]
            )))),
            'token_count': row['token_count']
        },
        axis=1
    )

    # Convert the column to a list

    unique_messages = {tuple(sorted(d.items())) for d in df_messages_filtered['message_and_participants'].tolist()}

    # Convert back to dictionaries
    unique_messages = [dict(t) for t in unique_messages]

    # Sort the list of dictionaries by the 'timestamp' key
    sorted_unique_messages = sorted(unique_messages, key=lambda x: x['timestamp'])

    filtered_messages = [m for m in sorted_unique_messages if start_date <= m['timestamp'] <= end_date]

    return filtered_messages


def get_top_100_token_dates(filtered_messages):
    # Create a DataFrame from filtered messages
    messages_df = pd.DataFrame(filtered_messages)
    
    # Group by date and sum the tokens
    messages_df['date'] = messages_df['timestamp'].apply(lambda x: x.date())
    date_token_sum = messages_df.groupby('date')['token_count'].sum()
    
    # Sort by the total token count and get the top 100
    top_100_dates = date_token_sum.sort_values(ascending=False).head(100)
    
    return top_100_dates


def main():
    parser = argparse.ArgumentParser(description='Process iMessages database.')
    parser.add_argument('start_date', type=str, help='Start date in YYYY-MM-DD format')
    parser.add_argument('end_date', type=str, help='End date in YYYY-MM-DD format')
    parser.add_argument('question', type=str, help='Question about texts')
    args = parser.parse_args()
    start_date = pd.Timestamp(args.start_date)
    end_date = pd.Timestamp(args.end_date)
    question = args.question
    prompt, token_count, filtered_messages = get_llm_prompt(question, start_date, end_date)
    top_100_dates = get_top_100_token_dates(filtered_messages)
    print(token_count)
    if token_count > 1000000:
        raise ValueError("Token count is greater than 1000000, please shorten the date range or use a bigger model.")
    print(prompt)
    for m in filtered_messages:
        print(m)

    genai.configure(api_key=os.environ["GOOGLE_AI_API_KEY"])

    model=genai.GenerativeModel(
        model_name="gemini-1.5-flash-exp-0827",
        system_instruction="The User will ask a Question and then give their Message History. You will carefully review the Question and the Message History and answer the question based on the message history to the best of your abilities.")

    safety_settings = [ {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}, {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}, {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"}, {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}, ]

    response = model.generate_content(
        [prompt],
        generation_config=genai.types.GenerationConfig(
            # Only one candidate for now.
            candidate_count=1,
            max_output_tokens=1000,
            temperature=1.0,
        ),
        safety_settings=safety_settings,
    )
    print(response)
    console = Console()
    md = Markdown(response.text)
    console.print(md)




#gemini-1.5-flash-exp-0827
#gemini-1.5-pro-exp-0827


if __name__ == "__main__":
    main()

