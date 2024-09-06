# iMessage-LLMLoader

## Setup

1. sudo cp /Users/${USER}/Library/Messages/chat.db ~/Downloads/imessages.db
2. On contacts app select all contacts then File > Export > Export vCard. and save as /Users/${USER}/Downloads/all_contacts_9_6_24.vcf
3. export GOOGLE_AI_API_KEY=<your api key>

## Usage

python imessage_llm_question.py "2024-08-07" "2024-09-07" "Give a day by day summary of the messages for each day. Keep the summary for each day concise."
