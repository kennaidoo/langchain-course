from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


def main() -> None:

    information = """
        Samuel Harris Altman (born April 22, 1985)[1] is an American entrepreneur, investor, and chief executive officer of OpenAI since 2019.
        [2] He is considered one of the leading figures of the AI boom.[3][4][5]
        Altman dropped out of Stanford University after two years and founded Loopt, a mobile social networking service, raising more than $30 million in venture capital. 
        In 2011, Altman joined Y Combinator, a startup accelerator, and was its president from 2014 to 2019.[6] 
        In 2019, he became CEO of OpenAI and oversaw the successful launch of ChatGPT in 2022. 
        He was ousted from the role by the company's board in 2023 due to a lack of confidence in his leadership, but was reinstated five days later following significant 
        backlash from employees and investors, after which a new board was formed.[4] 
        He has served as chairman of clean energy companies Helion Energy[7] and Oklo (until April 2025).[8] 
        As of December 2025, Altman's net worth is estimated at US$2.1 billion.[9] 
        In 2025, he was named as one of the "Architects of AI" for Time's Person of the Year.
    """
    summary_template = """
        Given the information {information} about a person I want you to create:
        1. A short summary
        2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], 
        template=summary_template
    )

    # llm = ChatOpenAI(temperature=0, model="gpt-5-nano-2025-08-07")
    llm = ChatOllama(
        temperature=0.1, 
        model="gpt-oss:20b", 
        validate_model_on_init=True, 
        reasoning=True
        )
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print(str(response.content))


if __name__ == "__main__":
    main()
