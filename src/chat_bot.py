import logging
import pandas as pd
import openai
from collections import Counter
import os
import fnmatch

from src.valid_index import ValidIndex
from src.file_tools import read_processed_regs_into_dataframe
from src.embeddings import get_ada_embedding, \
                           get_closest_nodes

class ExconManual():
    def __init__(self, log_file = ''):
        # Set up basic configuration first
        if log_file == '':
            logging.basicConfig(level=logging.INFO)
        else: 
            logging.basicConfig(filename=log_file, filemode='w', format='%(name)s - %(levelname)s - %(message)s')
        
        # Then get the logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)  # Set level for the logger

        # Read the regs into a dataframe and build the tree structure for the index
        excon_index_patterns = [
            r'^[A-Z]\.\d{0,2}',
            r'^\([A-Z]\)',
            r'^\((i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx|xxi|xxii|xxiii)\)',
            r'^\([a-z]\)',
            r'^\([a-z]{2}\)',
            r'^\((?:[1-9]|[1-9][0-9])\)',
        ]
        self.index_checker = ValidIndex(regex_list_of_indices=excon_index_patterns, exclusion_list=exclusion_list)
        self.df_excon = pd.read_csv("./inputs/excon_processed_manual.csv", sep="|", encoding="utf-8")
        

        # Load the definitions. 
        excon_definitions_and_embeddings_file = "./inputs/definitions_with_embeddings.parquet"
        self.df_definitions = pd.read_parquet(excon_definitions_and_embeddings_file, engine='pyarrow')

        excon_definitions_and_embeddings_file = "./inputs/definitions_insurance_with_embeddings.parquet"
        self.df_definitions_insurance = pd.read_parquet(excon_definitions_and_embeddings_file, engine='pyarrow')

        excon_definitions_and_embeddings_file = "./inputs/definitions_securities_with_embeddings.parquet"
        self.df_definitions_securities = pd.read_parquet(excon_definitions_and_embeddings_file, engine='pyarrow')

        
        # Load the section headings. 
        excon_section_headings_and_embeddings = "./inputs/section_headings_with_embeddings.parquet"
        self.df_sections = pd.read_parquet(excon_section_headings_and_embeddings, engine='pyarrow')
        # load the summary and questions
        section_summary_with_embeddings = "./inputs/summary_with_embedding.parquet"
        self.df_summary = pd.read_parquet(section_summary_with_embeddings, engine='pyarrow')
        section_questions_with_embeddings = "./inputs/questions_with_embedding.parquet"
        self.df_summary_questions = pd.read_parquet(section_questions_with_embeddings, engine='pyarrow')

        self.system_states = ["start", "no_relevant_embeddings", "requires_additional_sections","stuck"]
        self.system_state = self.system_states[0]
        self.messages = []
    #     self.msg = system_messages()


    def get_regulation_detail(self, node_str):
        return get_regulation_detail(node_str, self.df_excon, self.index_checker)

    def control_loop(self, user_context, threshold, model_to_use, temperature, max_tokens):
        if self.system_state == "struck":
            return "Unfortunately the system is in an unrecoverable state. Please restart the chat"
        elif self.system_state == "start":
            self.logger.info("Starting control loop with the query: " + user_context)
            self.messages.append({"role": "user", "content": user_context})
            df_definitions, df_search_sections = self.get_relevant_section_for_query_partial_toc(user_context, threshold)
            if len(df_definitions) + len(df_search_sections) == 0:
                self.system_state = "no_relevant_embeddings"
                self.logger.info("Unable to find any definitions or text related to this query")
                return "Unfortunately, there is no reference material in the database which can assist me to respond to your input. If you enter a query about the Exchange Control Manual, I will do my best to answer it."
            else:
                flag, response = self.resource_augmented_query(user_context, threshold, model_to_use, temperature, max_tokens, df_definitions, df_search_sections)
                if flag == "FAIL:":
                    self.logger.error("RAG did not return a response in the expected format")
                    self.system_state = "stuck"

                elif flag == "ANSWER:": # , "SECTION:", "NONE:"]
                    self.messages.append({"role": "assistant", "content": response})
                    self.system_state = "start"
                    return response
        else:
            self.logger.error("The system is in an unknown state")
            self.messages.append({"role": "assistant", "content": "The system is in an unknown state. Please restart the chat"})



    def get_relevant_section_for_query_partial_toc(self, user_context, threshold = 0.15):

        question_embedding = get_ada_embedding(user_context)

        definitions = []
        relevant_definitions = get_closest_nodes(self.df_definitions, embedding_column_name = "Embedding", question_embedding = question_embedding, n=4)
        for index, row in relevant_definitions.iterrows():
            definitions.append([row["cosine_dist"], row["Definition"], "all"])

        relevant_definitions = get_closest_nodes(self.df_definitions_insurance, embedding_column_name = "Embedding", question_embedding = question_embedding, n=4)
        for index, row in relevant_definitions.iterrows():
            definitions.append([row["cosine_dist"], row["Definition"], "insurance"])

        relevant_definitions = get_closest_nodes(self.df_definitions_securities, embedding_column_name = "Embedding", question_embedding = question_embedding, n=4)
        for index, row in relevant_definitions.iterrows():
            definitions.append([row["cosine_dist"], row["Definition"], "securities"])


        df_definitions = pd.DataFrame(definitions, columns = ["distance", "definition", "source"]).sort_values(by="distance", ascending=True)
        df_definitions["word_count"] = df_definitions["definition"].apply(num_tokens_from_string)
        df_definitions = df_definitions[df_definitions['distance'] < threshold]
        if len(df_definitions) > 0:
            self.logger.info("#################   Relevant Definitions       #################")
            for index, row in df_definitions.iterrows():
                self.logger.info(f'{row["distance"]:.4f}: ({row["source"]:>10}): ({row["word_count"]:>5}): {row["definition"]}')
        else:
            self.logger.info("#################   No relevant definitions found       #################")


        sections = []
        relevant_sections = get_closest_nodes(self.df_sections, embedding_column_name = "Embedding", question_embedding = question_embedding, n=4)
        for index, row in relevant_sections.iterrows():
            sections.append([row["cosine_dist"], row["section"], row["heading"], "heading"])

        relevant_section_summary = get_closest_nodes(self.df_summary, embedding_column_name = "Embedding", question_embedding = question_embedding, n=4)
        for index, row in relevant_section_summary.iterrows():
            sections.append([row["cosine_dist"], row["section"], row["text"], "summary"])

        relevant_section_questions = get_closest_nodes(self.df_summary_questions, embedding_column_name = "Embedding", question_embedding = question_embedding, n=4)
        for index, row in relevant_section_questions.iterrows():
            sections.append([row["cosine_dist"], row["section"], row["text"], "questions"])

        df_sections = pd.DataFrame(sections, columns = ["distance", "reference", "text", "source"]).sort_values(by="distance", ascending=True)
        result = df_sections[df_sections['distance'] < threshold]
        relevant_context = []
        if len(result) > 0:
            self.logger.info("#################   Relevant Sections       #################")
            for index, row in result.iterrows():
                self.logger.info(f'{row["distance"]:.4f}: {row["reference"]:>20}: {row["source"]:>15}: {row["text"]}')
                relevant_context.append(result.iloc[0]["reference"])
        else:
            self.logger.info("#################   No relevant sections found       #################")
        # Get the top result
        search_sections = []
        df_search_sections = pd.DataFrame()
        if len(result) > 0:
            self.logger.info(f'Top result: {result.iloc[0]["reference"]}')
            search_sections.append(result.iloc[0]["reference"])

            # 1) Get the mode
            mode_value_list = result['reference'].mode()
            mode_value = ""
            if len(mode_value_list) == 1:
                mode_value = mode_value_list[0]
                self.logger.info(f"Most common reference: {mode_value}")
                if mode_value not in search_sections:
                    search_sections.append(mode_value)
            else:
                self.logger.info("No unique mode")

            # 2) Get strings that occur multiple times along with their counts
            count_dict = Counter(result['reference'])
            repeated_items = {k: v for k, v in count_dict.items() if v > 1}
            # Remove the mode from repeated_items if it exists
            if mode_value in repeated_items:
                del repeated_items[mode_value]
            if (len(repeated_items) > 0):
                self.logger.info("References found that occour multiple times")
                for reference, count in repeated_items.items():
                    self.logger.info(f"Reference: {reference}, Count: {count}")
                    if reference not in search_sections:
                        search_sections.append(reference)
            
            df_search_sections = pd.DataFrame(search_sections, columns = ["reference"])
            df_search_sections["raw_text"] = df_search_sections["reference"].apply(self.get_regulation_detail)
            df_search_sections["word_count"] = df_search_sections["raw_text"].apply(num_tokens_from_string)

            if len(df_search_sections) > 0:
                self.logger.info("RAG Data:")
                for index, row in df_search_sections.iterrows():
                    self.logger.info(f'{row["word_count"]:>5}: {row["reference"]:>20}: {row["raw_text"]}')
            else:
                self.logger.info("No RAG Data.")
        else:
            self.logger.info("No results found within tollerance")
        return df_definitions, df_search_sections 


    def resource_augmented_query(self, user_context, threshold, model_to_use, temperature, max_tokens, df_definitions, df_search_sections):

        prefixes = ["ANSWER:", "SECTION:", "NONE:"]

        if len(df_definitions) + len(df_search_sections) > 0: # should always be the case as we check this in the control loop
            system_context = f"You are attempting to answer questions from an Authorised Dealer based only on the relevant documentation provided. You have only three options:\n\
1) Answer the question. If you do this, your must preface to response with the word '{prefixes[0]}'. If possible also provide a reference to the relevant documentation for the user to cross-check. Use this if you are sure about your answer.\n\
2) Request additional documentation. If, in the body of the relevant documentation, is a reference to another section of the document that is directly relevant, respond with the word '{prefixes[1]}' followed by the section reference which looks like A.1(A)(i)(aa). \n\
3) State '{prefixes[2]}' (and nothing else) in all cases where you are not confident about either of the first two options"
            if len(df_definitions) > 0:
                system_context = system_context + "\nPotentially relevant definition(s):"
                for index, row in df_definitions.iterrows():
                    system_context = system_context + "\n" + row["definition"]
            if len(df_search_sections) > 0:
                system_context = system_context + "\nPotentially relevant document section(s):"
                for index, row in df_search_sections.iterrows():
                    system_context = system_context + "\n" + row["raw_text"]

            self.logger.info("#################   Initial System Prompt       #################")
            self.logger.info("\n" + system_context)
            question_messages = [{"role": "system", "content": system_context}] + self.messages # don't change self.messages and don't add system_context to it

            total_tokens = num_tokens_from_string(system_context) + num_tokens_from_string(user_context)
            if total_tokens > 4000 and model_to_use!="gpt-3.5-turbo-16k":
                self.logger.warning("!!! NOTE !!! You have a very long prompt. Switching to the gpt-3.5-turbo-16k model")
                model_to_use = "gpt-3.5-turbo-16k"


            response = openai.ChatCompletion.create(
                                model=model_to_use,
                                temperature = temperature,
                                max_tokens = max_tokens,
                                messages = question_messages
                            )
            initial_response = response['choices'][0]['message']['content']

            for prefix in prefixes:
                if initial_response.startswith(prefix):
                    # Split the string into two parts: the prefix and the remaining text
                    return prefix, initial_response[len(prefix):]

            # now we need to recheck our work!
            self.logger.warning("Initial call did not create a resonse in the correct format. Retrying")
            self.logger.warning("The response was:")
            self.logger.warning(initial_response)
            despondent_user_context = f"Please check your answer and make sure your answer uses only one of the three permissible forms, {prefixes[0]}, {prefixes[1]} or {prefixes[2]}"
            despondent_user_messages = question_messages + [
                                        {"role": "assistant", "content": initial_response},
                                        {"role": "user", "content": despondent_user_context}]
                                        
            followup_response = openai.ChatCompletion.create(
                                model=model_to_use,
                                temperature = temperature,
                                max_tokens = max_tokens,
                                messages = despondent_user_messages
                            )
            followup_response_text = followup_response['choices'][0]['message']['content']
            for prefix in prefixes:
                if followup_response_text.startswith(prefix):
                    # Split the string into two parts: the prefix and the remaining text
                    return prefix, followup_response_text[len(prefix):]

        return "FAIL:", "The LLM was not able to return an acceptable answer. "



        # self.logger.info("#################   Initial AI Response       #################")
        # self.logger.info(initial_response)

        # is_valid_reference, modified_initial_response = self.response_is_a_reference(initial_response)
        # if is_valid_reference:
        #     self.logger.info(f"AI Response was a valid reference: {modified_initial_response}")
        #     return is_valid_reference, modified_initial_response    
        # else:
        #     # check to see if it returned a reference and the text
        #     s = extract_before_colon(initial_response)
        #     is_valid_reference, modified_initial_response = self.response_is_a_reference(s)
        #     if is_valid_reference:
        #         self.logger.info(f"Was able to extract a valid reference from this response. The value extracted is: {modified_initial_response}")
        #         return is_valid_reference, modified_initial_response    
        #     else:
        #         self.logger.info(f"AI Response did not contain a reference: {initial_response}")
        #         return is_valid_reference, initial_response

    # def get_relevant_section_for_query_full_toc(self, model_to_use, user_context):
    #     ### TODO: Move the ToC from the System to the User prompt
    #     return NotImplementedError("This needs to be updated so that the ToC is moved to the user prompt")        
    #     # self.logger.info("#################   Model for ToC query       #################")
    #     # self.logger.info("Model: " + model_to_use)

    #     # #msg = system_messages()
    #     # system_content = self.msg.system_check_against_toc + "\n" + self.tree_toc._list_node_children(self.tree_toc.root)
    #     # self.logger.info("#################   Initial System Prompt       #################")
    #     # self.logger.info(system_content)

    #     # self.logger.info("#################   Initial User Prompt       #################")
    #     # self.logger.info(user_context)

    #     # response = openai.ChatCompletion.create(
    #     #                     model=model_to_use,
    #     #                     temperature = 1.0,
    #     #                     max_tokens = 200,
    #     #                     messages=[
    #     #                         {"role": "system", "content": system_content},
    #     #                         {"role": "user", "content": user_context},
    #     #                     ]
    #     #                 )
    #     # initial_response = response['choices'][0]['message']['content']
    #     # self.logger.info("#################   Initial AI Response       #################")
    #     # self.logger.info(initial_response)

    #     # is_valid_reference, modified_initial_response = self.response_is_a_reference(initial_response)

    #     # return is_valid_reference, modified_initial_response


    # # use this method if the is_valid_reference flag is false in the "get_relevant_section_for_query" method
    # def confirm_relevant_section_for_query(self, model_to_use, user_context, initial_response):
    #     ### TODO: Update this so the ToC is in the user context
    #     return NotImplementedError("This needs to be updated so that the ToC is moved to the user prompt")        
    #     # is_valid_reference, modified_initial_response = self.response_is_a_reference(initial_response)
    #     # if is_valid_reference:
    #     #     required_section = modified_initial_response
    #     #     self.logger.info("The initial response contained only a valid reference. Moving forward with it")
    #     #     return is_valid_reference, modified_initial_response
    #     # else:
    #     #     self.logger.info("The initial response did not contain only a valid reference. Checking to see if a valid reference was provided but with additional text")
    #     #     system_content = self.msg.system_check_against_toc + "\n" + self.tree_toc._list_node_children(self.tree_toc.root)
    #     #     response = openai.ChatCompletion.create(
    #     #                         model=model_to_use,
    #     #                         temperature = 0.0,
    #     #                         max_tokens = 200,
    #     #                         messages=[
    #     #                             {"role": "system", "content": system_content},
    #     #                             {"role": "user", "content": user_context},
    #     #                             {"role": "assistant", "content": initial_response},
    #     #                             {"role": "user", "content": self.msg.system_check_ouput},
    #     #                         ]
    #     #                     )
    #     #     checked_response = response['choices'][0]['message']['content']

    #     #     is_valid_reference, modified_checked_response = self.response_is_a_reference(checked_response)
    #     #     if is_valid_reference:
    #     #         self.logger.info("Was able to extract the valid reference: " + modified_checked_response + " from the initial response. Moving forward with this reference")
    #     #         required_section = modified_checked_response
    #     #         return is_valid_reference, required_section
    #     #     else:
    #     #         error_msg = "Invalid value. Expected a valid section reference but did not get one. Consider phrasing the question differently"
    #     #         logging.error(error_msg)
    #     #         logging.error("Message from Chat: " + modified_checked_response)
    #     #         raise ValueError("Invalid value. Expected a valid section reference but did not get one. Check logs for detail")
    #     # return is_valid_reference, initial_response


    # def get_query_with_section_text(self, model_to_use, user_context, required_section):
    #     system_check_regs = self.msg.system_check_against_regulation
    #     self.logger.info("#################   First System Prompt to check regs        #################")
    #     self.logger.info(system_check_regs)

    #     self.logger.info("#################   Initial User Prompt       #################")
    #     relevant_sections_text = self.get_regulation_detail(required_section)
    #     user_prompt = f"### Question\n{user_context}\n### Regulations\n{relevant_sections_text}"
    #     self.logger.info(user_prompt)

    #     response = openai.ChatCompletion.create(
    #             model=model_to_use,
    #             temperature = 1.0,
    #             max_tokens = 500,
    #             messages=[
    #                 {"role": "system", "content": system_check_regs},
    #                 {"role": "user", "content": user_prompt}
    #             ]
    #         )
    #     initial_response_to_regulation = response['choices'][0]['message']['content']
    #     self.logger.info("#################   First AI Response to question with relevant regs        #################")
    #     self.logger.info(initial_response_to_regulation)

    #     if initial_response_to_regulation.startswith('ANSWER: '):
    #         self.logger.info('The initial response is flagged as an answer')
    #         return True, initial_response_to_regulation
    #     elif initial_response_to_regulation.startswith('SECTION: '):
    #         self.logger.info('The initial response is flagged as a request for more information')
    #         additional_section_requested = initial_response_to_regulation[len('SECTION: '):]
    #         is_valid_reference, modified_additional_section_requested = self.response_is_a_reference(additional_section_requested)
    #         if not is_valid_reference:
    #             self.logger.info('... but was not able to extract the refernce from it')
    #             return False, initial_response_to_regulation
    #         else:
    #             print('The LMM has requested additional regulation extracts in order to answer the question. Running a followup query ....')
    #             self.logger.info("#################   Followup User Prompt       #################")
    #             followup_section = self.get_regulation_detail(modified_additional_section_requested)
    #             user_prompt = user_prompt + "\n" + followup_section
    #             self.logger.info(user_prompt)
    #             response = openai.ChatCompletion.create(
    #                     model=model_to_use,
    #                     temperature = 1.0,
    #                     max_tokens = 500,
    #                     messages=[
    #                         {"role": "system", "content": system_check_regs},
    #                         {"role": "user", "content": user_prompt}
    #                     ]
    #                 )
    #             followup_response_to_regulation = response['choices'][0]['message']['content']
    #             if followup_response_to_regulation.startswith('ANSWER: '):
    #                 self.logger.info('The followup response is flagged as an answer')
    #                 return True, followup_response_to_regulation
    #             else:
    #                 self.logger.info('The followup response is NOT flagged as an answer')
    #                 return false, followup_response_to_regulation
            
    #     else:
    #         self.logger.info('The initial response is NOT flagged as an answer nor a request for more regulations')
    #         return False, initial_response_to_regulation



    # def followup_query():
    #     is_valid_reference, modified_response_to_regulation = self.response_is_a_reference(initial_response_to_regulation)
    #     if not is_valid_reference:
    #         self.logger.info("The AI response was not simply a request for more regulations. Checking to make sure it is not a request for more information but using the incorrect format")
    #         please_check = "Please check the assistant response. If it is a request for additional sections, please format the request so that the only part of the response is the section reference with no additional text. Each index in the section must be surronded by round brackets and there should be no white spaces. If it is not a request of additional sections, just repeat the response"
    #         response = openai.ChatCompletion.create(
    #                             model=model_to_use,
    #                             temperature = 0.0,
    #                             max_tokens = 200,
    #                             messages=[
    #                                 {"role": "system", "content": system_check_regs},
    #                                 {"role": "user", "content": user_context},
    #                                 {"role": "assistant", "content": initial_response_to_regulation},
    #                                 {"role": "user", "content": please_check},
    #                             ]
    #                         )
    #         checked_initial_response = response['choices'][0]['message']['content']
    #         # check again if this is a reference. If not, end here here by printing the response, if so proceed to next call
    #         is_valid_reference, modified_response_to_regulation = self.response_is_a_reference(checked_initial_response)
    #         if (is_valid_reference):
    #             self.logger.info("After further analysis, the AI requested additional information from section: " + modified_response_to_regulation)
    #         else:
    #             return("FINAL ANSWER: " + checked_initial_response)

    #     if is_valid_reference:
    #         self.logger.info("Adding the new section to the test to be sent to the model")
    #         additional_relevant_section = get_full_text_for_node(modified_response_to_regulation, self.df)    
    #         system_check_regs = system_check_regs + "\n" + additional_relevant_section
    #         self.logger.info("#################   Second System Prompt to check regs        #################")
    #         self.logger.info(system_check_regs)
    #         response = openai.ChatCompletion.create(
    #                             model=model_to_use,
    #                             temperature = 1.0,
    #                             max_tokens = 200,
    #                             messages=[
    #                                 {"role": "system", "content": system_check_regs},
    #                                 {"role": "user", "content": user_context}
    #                             ]
    #                         )
    #         second_response_to_regulation = response['choices'][0]['message']['content']
    #         self.logger.info("#################   Second AI Response to question with relevant regs        #################")
    #         self.logger.info(second_response_to_regulation)
    #         return("FINAL ANSWER: " + second_response_to_regulation)

