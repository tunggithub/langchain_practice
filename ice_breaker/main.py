from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from third_parties import scrape_linkedin_profile
from agents import lookup
import json

from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    summary_template = """
        given the {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["infomation"], template=summary_template
    )

    linkedin_profile_url = lookup(name="Andrew NG")
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    print(chain.run(information=linkedin_data))
