import os
import json
from openai import OpenAI
from datetime import datetime, time, timedelta
from dotenv import load_dotenv

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
load_dotenv()
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, ChatMessage

# --------------------------------------------------------------
# 1. Ask ChatGPT a Question
# --------------------------------------------------------------

completion = client.chat.completions.create(
    model="gpt-3.5-turbo-0613",
    messages=[
        {
            "role": "user",
            "content": "What are your hours of business?",
        },
])

response = completion.choices[0].message.content
print(response)

# --------------------------------------------------------------
# 2. Use OpenAI’s Function Calling Feature
# --------------------------------------------------------------

function_descriptions = [
    {
        "name": "get_business_hours",
        "description": "Check office opening hours before scheduling appointment",
        "parameters": {
            "type": "object",
            "properties": {
                "day": {
                    "type": "string",
                    "description": "Day of the week, e.g. Monday",
                },
                "open": {
                    "type": "string",
                    "description": "Office opening time, e.g. 9am",
                },
                "close": {
                    "type": "string",
                    "description": "Office closing time, e.g. 8pm",
                },
            },
            "required": ["day", "open", "close"],
        },
    },
    {
        "name": "is_business_open",
        "description": "Check if business is open",
        "parameters": {
            "type": "object",
            "properties": {
                "current_day": {
                    "type": "string",
                    "description": "Today's day, e.g. Monday",
                },
                "current_time": {
                    "type": "string",
                    "description": "Current time, e.g. 09:30",
                },
            },
            "required": ["current_day", "current_time"],
        },
    }
]

user_prompt = "Can you suggest an appointment today or tomorrow within opening and closing times for that day?"

completion = client.chat.completions.create(
    model="gpt-3.5-turbo-0613",
    messages=[{"role": "user", "content": user_prompt}],
    functions=function_descriptions,
function_call="auto")

# It automatically fills the arguments with correct info based on the prompt
# Note: the function does not exist yet, so ChatGPT output will be wrong!

response = completion.choices[0].message
print(response)

# --------------------------------------------------------------
# 3. Add a Function
# --------------------------------------------------------------

def get_business_hours(day=None):
    # Define office hours for each day of the week
    office_hours = {
        "Monday": {"open": "8:00", "close": "19:00"},
        "Tuesday": {"open": "8:00", "close": "19:00"},
        "Wednesday": {"open": "8:00", "close": "19:00"},
        "Thursday": {"open": "8:00", "close": "19:00"},
        "Friday": {"open": "8:00", "close": "19:00"},
        "Saturday": {"open": "9:00", "close": "16:00"},
        "Sunday": {"open": "10:00", "close": "15:00"},
    }

    # If no day is specified, return office hours for all days
    if day is None:
        return json.dumps(office_hours)
    # If a specific day is provided, return office hours for that day
    elif day in office_hours:
        return json.dumps(office_hours[day])
    else:
        return "Invalid day"

  
office_hours = get_business_hours()

# Use the LLM output from above to manually call the function
# The json.loads function converts the string to a Python dictionary

day = json.loads(response.function_call.arguments).get("day")
params = json.loads(response.function_call.arguments)
type(params)

print(day)
print(params)

# Call the function with arguments

chosen_function = eval(response.function_call.name)
office_hours=chosen_function()
print(office_hours)

def is_business_open(current_day=None, current_time=None):
    # If current_day and current_time are not provided, fetch them
    if current_day is None or current_time is None:
        now = datetime.now()
        current_day = now.strftime("%A")
        current_time = now.time()
        
        
    office_hours_json = get_business_hours()
    office_hours = json.loads(office_hours_json)

    # Get the office hours for the current day
    opening_time_str, closing_time_str = office_hours[current_day]["open"], office_hours[current_day]["close"]
    opening_time = datetime.strptime(opening_time_str, "%H:%M").time()
    closing_time = datetime.strptime(closing_time_str, "%H:%M").time()

    # Check if the current time is within the office hours
    if opening_time <= current_time <= closing_time:
        return "The office is open."
    else:
        return "The office is closed."

print(is_business_open())
print(datetime.now())

# --------------------------------------------------------------
# 4. Add function result to the prompt for a final answer
# --------------------------------------------------------------

# The key is to add the function output back to the messages with role: function
# Call the function to get office hours

office_hours = get_business_hours()
# Confirm the content of the office hours variable
print(office_hours)

# Ensure that office_hours is not None and is properly formatted JSON
if office_hours:
    print("office_hours is not None")
    try:
        office_hours_json = json.loads(office_hours)
        print("office_hours is valid JSON")
    except json.JSONDecodeError as e:
        print("Error decoding office_hours JSON:", e)
else:
    print("office_hours is None")

# Check the completion request to ensure office_hours is passed correctly
print("Completion request:")
print({
    "role": "function",
    "name": response.function_call.name,
    "content": office_hours
})

# Make sure the completion request is created with correct parameters
second_completion = client.chat.completions.create(
    model="gpt-3.5-turbo-0613",
    messages=[
        {"role": "user", "content": user_prompt},
        {"role": "function", "name": response.function_call.name, "content": office_hours},
    ],
    functions=function_descriptions,
)

# Check the response from the completion request
print("Completion response:")
print(second_completion)

# Manually call the function with different inputs
print("Testing get_business_hours() function:")

# Test case 1: Monday
print("Testing for Monday:")
monday_hours = get_business_hours("Monday")
print("Monday office hours:", monday_hours)

# Test case 2: Saturday
print("Testing for Saturday:")
saturday_hours = get_business_hours("Saturday")
print("Saturday office hours:", saturday_hours)

# Test case 3: Sunday
print("Testing for Sunday:")
sunday_hours = get_business_hours("Sunday")
print("Sunday office hours:", sunday_hours)

# Test case 3: Sunday
print("Testing for Sunday:")
wednesday_hours = get_business_hours("Wednesday")
print("Wednesday office hours:", wednesday_hours)

# Test case 3: What time?
print("Testing for open:")
current_time = is_business_open("current_time")
print("Is business open?:", current_time)

# --------------------------------------------------------------
# Include Multiple Functions
# --------------------------------------------------------------

# Expand on function descriptions (3 functions)

function_descriptions_multiple = [
    {
        "name": "schedule_appointment",
        "description": "Suggest an appointment during office hours",
        "parameters": {
            "type": "object",
            "properties": {
                "loc_origin": {
                    "type": "string",
                    "description": "The departure airport, e.g. DUS",
                },
                "loc_destination": {
                    "type": "string",
                    "description": "The destination airport, e.g. HAM",
                },
            },
            "required": ["loc_origin", "loc_destination"],
        },
    },
    {
        "name": "select_car",
        "description": "TBC",
        "parameters": {
            "type": "object",
            "properties": {
                "loc_origin": {
                    "type": "string",
                    "description": "The departure airport, e.g. DUS",
                },
                "loc_destination": {
                    "type": "string",
                    "description": "The destination airport, e.g. HAM",
                },
                "datetime": {
                    "type": "string",
                    "description": "The date and time of the flight, e.g. 2023-01-01 01:01",
                },
                "airline": {
                    "type": "string",
                    "description": "The service airline, e.g. Lufthansa",
                },
            },
            "required": ["loc_origin", "loc_destination", "datetime", "airline"],
        },
    },
    {
        "name": "file_complaint",
        "description": "File a complaint as a customer",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name of the user, e.g. John Doe",
                },
                "email": {
                    "type": "string",
                    "description": "The email address of the user, e.g. john@doe.com",
                },
                "text": {
                    "type": "string",
                    "description": "Description of issue",
                },
            },
            "required": ["name", "email", "text"],
        },
    },
]

print(function_descriptions_multiple)

def ask_and_reply(prompt):
    """Give LLM a given prompt and get an answer."""

    completion = client.chat.completions.create(model="gpt-3.5-turbo-0613",
    messages=[{"role": "user", "content": prompt}],
    # add function calling
    functions=function_descriptions_multiple,
    function_call="auto")

    output = completion.choices[0].message
    return output


# Scenario 1: Are you open now?

user_prompt = "Are you open right now?"
print(ask_and_reply(user_prompt))

# Get info for the next prompt

day = json.loads(response.function_call.arguments).get("day")
close = json.loads(response.function_call.arguments).get("close")
chosen_function = eval(response.function_call.name)
office_hours = chosen_function(day)

print()
print()
print()

flight_datetime = json.loads(flight).get("datetime")
flight_airline = json.loads(flight).get("airline")

print(flight_datetime)
print(flight_airline)

# Scenario 2: Book appointment for Friday

user_prompt = f"I want to book an appointment on Saturday"
print(ask_and_reply(user_prompt))

# Scenario 3: Book appointment for next week

user_prompt = "I am on holiday, can I book a call next week?"
print(ask_and_reply(user_prompt))

# --------------------------------------------------------------
# Make It Conversational With Langchain TBD
# --------------------------------------------------------------

llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0)

# Start a conversation with multiple requests

user_prompt = """
This is Jane Doe. I am an unhappy customer that wants you to do several things.
First, I need to know when's the next available appointment.
Please proceed to book that appointment for me as soon as possible.
Also, I want to file a complaint about my missed flight. It was an unpleasant surprise. 
Email me a copy of the complaint to jane@doe.com.
Please give me a confirmation after all of these are done.
"""

# Returns the function of the first request (get_flight_info)

first_response = llm.predict_messages(
    [HumanMessage(content=user_prompt)], functions=function_descriptions_multiple
)

print(first_response)

# Returns the function of the second request (book_appointment)
# It takes all the arguments from the prompt but not the returned information

second_response = llm.predict_messages(
    [
        HumanMessage(content=user_prompt),
        AIMessage(content=str(first_response.additional_kwargs)),
        AIMessage(
            role="function",
            additional_kwargs={
                "name": first_response.additional_kwargs["function_call"]["name"]
            },
            content=f"Completed function {first_response.additional_kwargs['function_call']['name']}",
        ),
    ],
    functions=function_descriptions_multiple,
)

print(second_response)

# Returns the function of the third request (file_complaint)

third_response = llm.predict_messages(
    [
        HumanMessage(content=user_prompt),
        AIMessage(content=str(first_response.additional_kwargs)),
        AIMessage(content=str(second_response.additional_kwargs)),
        AIMessage(
            role="function",
            additional_kwargs={
                "name": second_response.additional_kwargs["function_call"]["name"]
            },
            content=f"Completed function {second_response.additional_kwargs['function_call']['name']}",
        ),
    ],
    functions=function_descriptions_multiple,
)

print(third_response)

# Conversational reply at the end of requests

fourth_response = llm.predict_messages(
    [
        HumanMessage(content=user_prompt),
        AIMessage(content=str(first_response.additional_kwargs)),
        AIMessage(content=str(second_response.additional_kwargs)),
        AIMessage(content=str(third_response.additional_kwargs)),
        AIMessage(
            role="function",
            additional_kwargs={
                "name": third_response.additional_kwargs["function_call"]["name"]
            },
            content=f"Completed function {third_response.additional_kwargs['function_call']['name']}",
        ),
    ],
    functions=function_descriptions_multiple,
)

print(fourth_response)