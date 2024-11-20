# The first thing we will do is visit the link we were giving to access the colab notebook that 
#the instructtor has for us.

# This is the link (https://ai.google.dev/tutorials/python_quickstart?authuser=1).

# Once we use the link and we are inside the google colab notebook we will find the get api key tab and 
#push it.

# This will begin the process os getting our API Key setup for Gemini.

#  If we want to talk to our Model we will need this API Key to make that possible.

# Now let's start working with our notebook.

# The first thing we will do inside our colab note book is install google generative AI package.

# This is the package we will be using.

!pip install -q -U google-generativeai

# This is the library we will be using.

# The first file we will be creating is our requirements file.

# Now that our requirement file is created we will pip install our google generative ai package.

# Now we have our first item to add to our requirements file. This will be the google-generativeai package
#we just installed.

# The next item we will add to our requirements file is the streamlit library.

# The reason we will be using streamlit is for our frontend portion of the project.

# The next thing we will do is access our cmd line from the terminal so we can create an environment.

# This is the command we will enter into our cmd line to do that.

# conda create -p venv python==3.9 -y

# Once we create our environment, we can either select it for the workspace folder or activate it.

# To activate our environment we can use the command on our cmd line.

# conda activate venv/ (this is what we named our environment).

# Now our environment is created and activated.

# Next we will start importing our libraries into our colab notebook.

# Note: google-generativeai library - All the functionality (let it be text, images, videos, or audio) of our 
#model will come from this library.

# The next step will be to initialize our API Key and save it to a variable inside our colab notebook.

# We should keep in mind that the API Key we created will be the same one used in our end to end project.

# The variable that we will be storing our API Key is GOOGLE_API_KEY.

# Because we want to secure our api key, we will create a dot env file (.env)

# This dot env file is where we are going to copy and paste our API Key to secure it.

# Once that is done we can move on.

# This is the code that the instructor is running in his colab notebook.

import pathlib
import textwrap

import google.generativeai as genai

# Used to securely store our api key 
from google.colab import userdata

api_key = ""

from IPython.display import display
from IPython.display import Markdown

# Then we will use to this code inside our colab notebook to help us modify our text. 

# This code will convert our text into mark down.

def to_markdown(text):
    text = text.replace("*", " *")
    return Markdown(textwrap.indent(text, "> ", predicate=lambda _:True)) 


# Note: We commented out this piece of code in our google colab because we are going to use our variable
#to store our API Key in. ( GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY') )

# Since we'll be using this piece of code( genai.configure(api_key=GOOGLE_API_KEY) ), we need to configure
#our API Key.

# This is how we will do that.

genai.configure(api_key=key)

# We set the configuration of our api key to equal the variable we have our api key stored in, which is key.

# Now our API Key is configured.

# Even though we have our API key displayed in our colab notebook, this isn't good practice.

# That's why we created our .env file.

# Now we can move forward.

# Let's look at the genai.config method.

# This method will provide us with two important models.

# These are the gemini-pro, and gemini-pro-vision models.

# gemini-pro: Optimized for text-only prompts.

# gemini-pro-vision: Optimized for text-and-images prompts.

# Here is more code from the instructors version of the quickstart colab notebook.

for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)


# Before we move forward, let's discuss some of the usecases of using these LLM models.

#   Usecases:

# : Extracting and utlizing data provided.
# : Automating specific task.
# : Modifying existing data.
# : Text Generation.
# : Chat / Q and A.

# Now that we have an understanding of usecases, let's move forward.

# The next thing we will do is install our requirements from our requirements.txt file.

# Inside of our environment we will run this command in the terminal (pip install -r requirements.txt)

# This is how it will look on our command line.

pip install -r requirements.txt

# Once our requirements are installed we can move forward.

# The next thing we will do is create our app.py file.

# Now let's get an understanding of what we'll specifically be doing.

# What is our plan or architecture of the code flow we will be creating.

# The first thing we will do is upload an image.

# Then we will write our own custom prompt.

# The prompt is where we will ask our questions.

# Then we will have an output.

# The output let's us that the image was built to certain specifications.

# Now let's understand the architecture.

# As soon as we upload the image, we will take the image and convert it into bytes.

# Then we will retrieve the image info.

# The next thing we will do is take the image info and add it with our prompt.

# The image info + prompt will get sent as a whole to our Gemini-Pro Large Lanuage Model.

# Once we are inside the Google-Pro LLM, it will look for two pieces of important information.

# These are the Prompt, and the Image.

# The Gemini-Pro has an internal OCR functionality.

# This means that it will try to compare the two pieces of information to try and obtain an output.

# The output will provide us with the answer to our prompt, which is our actual output.

# So this is what we are specifically going to do.

# We are going to design our model.

# We will write the code for our Prompt.

# And finally we will get our result in the form of an output.

# Now that we have an idea of our architecture, we can move forward and start building step by step.

# Now we will go back to the app.py file we craeted.

# The first thing we will add to our app.py file is comment that tells what type of actions we will be
#taking.

# This is the comment will go into our app.py file. # Invoice Extractor

# The next we will be doing inside our app.py file is initializing our API Key from our .env file.

# This is the code we will write in our app.py file to access our API Key in this file.

from dotenv import load_dotenv

# What this load_dotenv will do for us is it we help us load all of our environment areas.

# But to have access to these features we will need to install it by adding it to our requirements file.

# Thsis is what we will be adding to our requirements file.

python-dotenv

# If we have already pip installed our requirements file before this we will need to reinstall the 
#requirements file so that the update packages may get installed.

# After our package is installed we can move forward inside the app.py file.

# The next piece of code we will be adding is the load_dotenv method.

# This is the code we will be adding.

load_dotenv() # This will load all of the environments from the .env file.

# The next thing we will do inside of our app.py file is import the streamlit package.

# This is how we will do that.

import streamlit as st

# Because we need call our environment variables, the next thing we will do is import the os
#package.

# This is how we will do that.

import os

# And because we're using images, the next package we are going to import is Image package from
#the PIL library.

# This is how we will do that.

from PIL import Image

# This Image package will actualy help us get the information from our images.

# The next thing we will do is import the google.generativeai as genai

# This is what that will look like in our app.py file

import google.generativeai as genai

# We need to load our API Key, but first we must configure it.

# This is hw we will do that.

## Configure API Key

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# The next step we will take is to write a function to load the Gemini Pro Vision Model.

# This is how we will do that.

## Function to load Gemini Pro Vision Model

def get_gemini_response(input, image, prompt):
    pass

# Inside our get gemini function, we are going to pass (input, image[0], prompt) as a parameters.

# Then we will go ahead and call our model, which is the variable that is storing our Generative
#Model function.

# Inside our Generative Model is where we are going to call the model verison of gemini we are using
#by pssing it as a parameter.

# Basically when we call the function this way we are loading our model, So we can use a message to 
#ourselves to remember.

# This is the message we will leave ourselves, (loading Gemini model).

# After loading our model we will want to get the response.

# The response will be the variable that holds our model dot generate content function.

# The model dot generate content function will take as a parameter a list.

# This list take three items. ([input, image[0], prompt]).

# Since the images will be returned as a list we want to index them with the position 0, so 
#that the list starts from the beginning.

# Once we get this information we will want to return the response.

# There will be a parametter inside of the returned response, which will be text.

def get_gemini_response(input, image, prompt):
    # loading Gemini model
    model = genai.GenerativeModel("gemini-pro-vision")
    response = model.generate_contetnt([input, image[0], prompt])
    return response.text

# We will be getting all of this information from our Gemini Model.

# What we are doing in this particular function is creating a function which will load a gemini
#ai model, and then the generative ai model will take the input and give us a response.

# The next thing we will do is write a function to get our image.

# This is the code we will use to write our function.

def input_image_setup():
    pass

# This function will take as a parameter (uploaded_file)

# Uploaded image represents whatever image will be using.

# Now that we have that, we can define our function.

def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type, # Get the mimie type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No File Uploaded")
    
# So what we are doing here is we're taking an image, and converting that image into bytes.

# The information we get from the bytes data will come in two categories, mimie_type and data.

# Then we will be returning the image_parts.

# If we don't have any information to return, we will raise an exception.


# This is the second task we have completed and now we can move on the third task.

# The task we will work on initializing our streamlit app.

# This is the code that will helpp us do that.

## Initialize Streamlit App

st.set_page_config(page_title = "Invoice Extractor")

st.header("Gemini Application")
input = st.text_input("Input Prompt: ", key = "input")
uploaded_file = st.file_uploader("Choose an image...", type = ["jpg", "jpeg", "png"])
image = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption = "Uploaded Image.", use_column_width = True)

submit = st.button("Tell Me About The Invoice") # We created a submit button for our streamlit app

# Finally, the next piece we need to add to our app.py file is our prompt.

# This prompt will tell our model how we want and expect it to behave.

# This is the promptt we will use for our model.

input_prompt = """
You are an expert in understanding invoices. You will
recieve images as invoices and you will be expected to
answer questions correctly and honestly based on the input images.
"""

# These are the instructions and guidelines we expect our model to behave by.

# Finally, we will program our submit button.

## If Submit Button is clicked

if submit:
    image_data = input_image_setup(uploaded_file)
    response = get_gemini_response(input_prompt, image_data, input)

    # The next thing we want to do is display our response.
    # This is method we will use to do that, plus it's parameter/text.
    st.subheader("The Response is")
    # This is how we will display the actual response.
    st.write(response)


# With that, our three functions are built, and our project is done. 


# Timestamp 25:31:00