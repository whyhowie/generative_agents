"""
Provider-agnostic LLM wrapper functions used across the simulation.
"""
import json
import time

from persona.prompt_template.llm_gateway import llm_request, llm_embedding


def temp_sleep(seconds=0.1):
  time.sleep(seconds)


def llm_single_request(prompt, model=None, params=None):
  temp_sleep()
  return llm_request(prompt, model=model, params=params)


def llm_safe_generate_response(prompt, 
                               example_output,
                               special_instruction,
                               repeat=3,
                               fail_safe_response="error",
                               func_validate=None,
                               func_clean_up=None,
                               verbose=False,
                               model=None,
                               params=None): 
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("LLM PROMPT")
    print (prompt)

  for i in range(repeat): 
    try: 
      curr_response = llm_request(prompt, model=model, params=params).strip()
      end_index = curr_response.rfind('}') + 1
      curr_response = curr_response[:end_index]
      curr_response = json.loads(curr_response)["output"]

      if func_validate(curr_response, prompt=prompt): 
        return func_clean_up(curr_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_response)
        print (curr_response)
        print ("~~~~")

    except: 
      pass

  return False


def llm_safe_generate_response_old(prompt, 
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False,
                                   model=None,
                                   params=None): 
  if verbose: 
    print ("LLM PROMPT")
    print (prompt)

  for i in range(repeat): 
    try: 
      curr_response = llm_request(prompt, model=model, params=params).strip()
      if func_validate(curr_response, prompt=prompt): 
        return func_clean_up(curr_response, prompt=prompt)
      if verbose: 
        print (f"---- repeat count: {i}")
        print (curr_response)
        print ("~~~~")

    except: 
      pass
  print ("FAIL SAFE TRIGGERED") 
  return fail_safe_response


def GPT_request(prompt, gpt_parameter): 
  temp_sleep()
  params = {
    "temperature": gpt_parameter.get("temperature"),
    "max_tokens": gpt_parameter.get("max_tokens"),
    "top_p": gpt_parameter.get("top_p"),
    "frequency_penalty": gpt_parameter.get("frequency_penalty"),
    "presence_penalty": gpt_parameter.get("presence_penalty"),
    "stream": gpt_parameter.get("stream"),
    "stop": gpt_parameter.get("stop"),
  }
  return llm_request(prompt, model=gpt_parameter.get("engine"), params=params)


def generate_prompt(curr_input, prompt_lib_file): 
  if type(curr_input) == type("string"): 
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):   
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt: 
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


def safe_generate_response(prompt, 
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False): 
  if verbose: 
    print (prompt)

  for i in range(repeat): 
    curr_response = GPT_request(prompt, gpt_parameter)
    if func_validate(curr_response, prompt=prompt): 
      return func_clean_up(curr_response, prompt=prompt)
    if verbose: 
      print ("---- repeat count: ", i, curr_response)
      print (curr_response)
      print ("~~~~")
  return fail_safe_response


def get_embedding(text, model=None):
  text = text.replace("\n", " ")
  if not text: 
    text = "this is blank"
  return llm_embedding(text, model=model)




















