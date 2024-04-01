
from ctransformers import AutoModelForCausalLM
import time

duration = []
with open("ctransformers.txt", 'w') as file:
    llm = AutoModelForCausalLM.from_pretrained("model\\mistral-7b-instruct-v0.2.Q8_0.gguf")
    for i in range(50):
        start_time = time.time()
        answer_1 = llm("Convert the numerical values in text form to numerical form in this sentence: \"The dimensions are four hundred and sixty three, one thousand eight hundred, ninety seven The phases are oil and water\"")
        print(answer_1)
        duration.append(time.time() - start_time)
        for a in duration:
            file .write(str(a)+ " ")
    print("The end")
with open("ctransformers2.txt", 'w') as file:
    llm = AutoModelForCausalLM.from_pretrained("model\\mistral-7b-instruct-v0.2.Q8_0.gguf")   
    for i in range(50):
        start_time = time.time()
        answer_1 = llm("In the following sentnece a text is enclosed between two commas, if a text contains numbers in text form Convert them to numerical digits form in place of the text: \"DIMENS COMMA four hundred and sixty three comma one thousand eight hundred comma ninety seven comma  phases comma oil comma water\"")
        print(answer_1)
        duration.append(time.time() - start_time)
        for a in duration:
            file .write(str(a)+ " ")
    print("The end")
    
