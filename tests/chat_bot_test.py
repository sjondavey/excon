import pandas as pd

import importlib
import src.chat_bot
importlib.reload(src.chat_bot)
from src.chat_bot import ExconManual

class TestExconManual:
    excon = ExconManual(log_file='')
    excon_test = ExconManual(log_file='',input_folder = "./inputs_test/")
    include_calls_to_api = True
        

    def test_construction(self):
        assert True

    def test_user_provides_input(self):
        # check the response if the system is stuck
        self.excon_test.system_state = self.excon_test.system_states[3] #stuck
        user_context = "How much money can an individual take offshore in any year?"
        self.excon_test.user_provides_input(user_context, 
                                       threshold = 0.15, 
                                       model_to_use="gpt-3.5-turbo", 
                                       temperature = 0, 
                                       max_tokens = 200)
        assert self.excon_test.messages[-1]["role"] == "assistant"
        assert self.excon_test.messages[-1]["content"] == self.excon_test.assistant_msg_stuck
        assert self.excon_test.system_state == self.excon_test.system_states[3]

        # check the response if the system is in an unknown state
        self.excon_test.system_state = "random state not in list"
        user_context = "How much money can an individual take offshore in any year?"
        self.excon_test.user_provides_input(user_context, 
                                       threshold = 0.15, 
                                       model_to_use="gpt-3.5-turbo", 
                                       temperature = 0, 
                                       max_tokens = 200)
        assert self.excon_test.messages[-1]["role"] == "assistant"
        assert self.excon_test.messages[-1]["content"] == self.excon.assistant_msg_unknown_state

        # check the response if there are no relevant documents
        self.excon_test.system_state = self.excon_test.system_states[0] # rag
        user_context = "How much money can an individual take offshore in any year?"
        self.excon_test.user_provides_input(user_context, 
                                       threshold = 0.15, 
                                       model_to_use="gpt-3.5-turbo", 
                                       temperature = 0, 
                                       max_tokens = 200)
        assert self.excon_test.messages[-1]["role"] == "assistant"
        assert self.excon_test.messages[-1]["content"] == self.excon_test.assistant_msg_no_data
        #assert self.excon_test.system_state == "no_relevant_embeddings"
        assert self.excon_test.system_state == self.excon_test.system_states[0] # rag

        # test the workflow if the system answers the question as hoped
        self.excon_test.system_state = self.excon_test.system_states[0] # rag
        user_context = "Who can trade gold?" # there are hits in the KB for this
        testing = True # don't make call to openai API, use the canned response below
        flag = "ANSWER:"
        response = "The acquisition of gold for legitimate trade purposes, such as by manufacturing jewellers or dentists, is subject to the approval of the South African Diamond and Precious Metals Regulator. After receiving such approval, a permit must be obtained from SARS, which will allow the permit holder to approach Rand Refinery Limited for an allocation of gold. The holders of gold, having received the necessary approvals, are exempt from certain provisions of Regulation 5(1). (Reference: C. Gold (C)(i))"
        manual_responses_for_testing = [flag + response]
        self.excon_test.user_provides_input(user_context, 
                                       threshold = 0.15, 
                                       model_to_use="gpt-3.5-turbo", 
                                       temperature = 0, 
                                       max_tokens = 200,
                                       testing = testing,
                                       manual_responses_for_testing = manual_responses_for_testing)
        assert self.excon_test.messages[-1]["role"] == "assistant"
        assert self.excon_test.messages[-1]["content"].strip() == response
        assert self.excon_test.system_state == self.excon_test.system_states[0] # rag

        # test the workflow if the system cannot find useful content in the supplied data
        self.excon_test.system_state = self.excon_test.system_states[0] # rag
        user_context = "Who can trade gold?" # there are hits in the KB for this
        testing = True # don't make call to openai API, use the canned response below
        flag = "NONE:"
        response = "None of the supplied documentation was relevant"
        manual_responses_for_testing = []
        manual_responses_for_testing.append(flag + response)
        self.excon_test.user_provides_input(user_context, 
                                       threshold = 0.15, 
                                       model_to_use="gpt-3.5-turbo", 
                                       temperature = 0, 
                                       max_tokens = 200,
                                       testing = testing,
                                       manual_responses_for_testing = manual_responses_for_testing)
        assert self.excon_test.messages[-1]["role"] == "assistant"
        assert self.excon_test.messages[-1]["content"].strip() == self.excon.assistant_msg_no_relevant_data
        assert self.excon_test.system_state == self.excon_test.system_states[0] # rag

        # test the workflow if the system needs additional content
        self.excon_test.system_state = self.excon_test.system_states[0] # rag
        user_context = "Who can trade gold?" # there are hits in the KB for this
        testing = True # don't make call to openai API, use the canned response below
        flag = "SECTION:"
        response = "C.(C)"
        manual_responses_for_testing = []
        manual_responses_for_testing.append(flag + response)

        # now the response once it has received the additional data
        flag = "ANSWER:"
        response = "The acquisition of gold for legitimate trade purposes, such as by manufacturing jewellers or dentists, is subject to the approval of the South African Diamond and Precious Metals Regulator. After receiving such approval, a permit must be obtained from SARS, which will allow the permit holder to approach Rand Refinery Limited for an allocation of gold. The holders of gold, having received the necessary approvals, are exempt from certain provisions of Regulation 5(1). (Reference: C. Gold (C)(i))"        
        manual_responses_for_testing.append(flag + response)

        self.excon_test.user_provides_input(user_context, 
                                       threshold = 0.15, 
                                       model_to_use="gpt-3.5-turbo", 
                                       temperature = 0, 
                                       max_tokens = 200,
                                       testing = testing,
                                       manual_responses_for_testing = manual_responses_for_testing)
        assert self.excon_test.messages[-1]["role"] == "assistant"
        assert self.excon_test.messages[-1]["content"].strip() == response
        assert self.excon_test.system_state == self.excon_test.system_states[0] # rag

        # test what happens if the LLM does not listen to instructions and returns something random
        self.excon_test.system_state = self.excon_test.system_states[0] # rag
        user_context = "Who can trade gold?" # there are hits in the KB for this
        testing = True # don't make call to openai API, use the canned response below
        response = "None of the supplied documentation was relevant"
        manual_responses_for_testing = []
        manual_responses_for_testing.append(response)
        manual_responses_for_testing.append(response) # need to add it twice when checking this branch
        self.excon_test.user_provides_input(user_context, 
                                       threshold = 0.15, 
                                       model_to_use="gpt-3.5-turbo", 
                                       temperature = 0, 
                                       max_tokens = 200,
                                       testing = testing,
                                       manual_responses_for_testing = manual_responses_for_testing)
        assert self.excon_test.messages[-1]["role"] == "assistant"
        assert self.excon_test.messages[-1]["content"].strip() == self.excon_test.assistant_msg_llm_not_following_instructions
        assert self.excon_test.system_state == self.excon_test.system_states[3] # stuck

        # Now all the test with additional sections requested


    def test_resource_augmented_query(self):
        user_context = "Who can trade gold?"
        relevant_definitions, relevant_sections = self.excon_test.similarity_search(user_context)
        # If there are no messages in the queue, we should get an error
        flag, response = self.excon_test.resource_augmented_query(model_to_use="gpt-3.5-turbo", 
                                                                temperature = 0, 
                                                                max_tokens = 200, 
                                                                df_definitions = relevant_definitions, 
                                                                df_search_sections = relevant_sections)
        assert flag == "stuck"
        assert response == "stuck"

        # Add a message to the queue and give the LLM relevant data from which to answer the question
        self.excon_test.system_state = "rag"
        self.excon_test.messages = [{"role": "user", "content": user_context}]
        # NOTE: I am not going to test the openai api call. I am going to use 'testing' mode with canned answers
        testing = True
        manual_responses_for_testing = []
        manual_responses_for_testing.append("ANSWER: test to see what happens when if the API believes it successfully answered the question with the resources provided")
        flag, response = self.excon_test.resource_augmented_query(model_to_use="gpt-3.5-turbo", 
                                                                temperature = 0, 
                                                                max_tokens = 200, 
                                                                df_definitions = relevant_definitions, 
                                                                df_search_sections = relevant_sections,
                                                                testing = testing,
                                                                manual_responses_for_testing = manual_responses_for_testing)
        assert flag == self.excon_test.rag_prefixes[0] # "ANSWER:"
        # Also check that nothing happened to the internal message stack
        assert len(self.excon_test.messages) == 1
        if self.include_calls_to_api: # also test it with a call to the API
            flag, response = self.excon_test.resource_augmented_query(model_to_use="gpt-3.5-turbo", 
                                                                temperature = 0, 
                                                                max_tokens = 200, 
                                                                df_definitions = relevant_definitions, 
                                                                df_search_sections = relevant_sections)
            assert flag == self.excon_test.rag_prefixes[0] # "ANSWER:"
            # Also check that nothing happened to the internal message stack
            assert len(self.excon_test.messages) == 1

        # Check that if the question and reference data mismatch, the system returns a NONE: value
        self.excon_test.messages = [{"role": "user", "content": "How much money can an individual take offshore in any year?"}]
        testing = True
        manual_responses_for_testing = []
        manual_responses_for_testing.append("NONE: test to see what happens when if the API believes it cannot answer the question with the resources provided")
        flag, response = self.excon_test.resource_augmented_query(model_to_use="gpt-3.5-turbo", 
                                                                temperature = 0, 
                                                                max_tokens = 200, 
                                                                df_definitions = relevant_definitions, 
                                                                df_search_sections = relevant_sections,
                                                                testing = testing,
                                                                manual_responses_for_testing = manual_responses_for_testing)
        assert flag == self.excon_test.rag_prefixes[2] # "NONE:"
        # Also check that nothing happened to the internal message stack
        assert len(self.excon_test.messages) == 1
        
        
        # Check the "SECTION" branch
        question = "Can you list the dispensations necessary for a rand facility to a non-resident exceed 6 months?"
        self.excon_test.messages = [{"role": "user", "content": question}]
        testing = True
        manual_responses_for_testing = []
        manual_responses_for_testing.append("SECTION: test to see what happens when if the API believes it needs information from another section")

        flag, response = self.excon_test.resource_augmented_query(model_to_use="gpt-3.5-turbo", #"gpt-4", 
                                                        temperature = 0, 
                                                        max_tokens = 200, 
                                                        df_definitions = relevant_definitions, 
                                                        df_search_sections = relevant_sections,
                                                        testing = testing,
                                                        manual_responses_for_testing = manual_responses_for_testing)
        assert flag == self.excon_test.rag_prefixes[1] # "SECTION:"
        # Also check that nothing happened to the internal message stack
        assert len(self.excon_test.messages) == 1

        # Check the despondent user branch
        self.excon_test.messages = [{"role": "user", "content": "How much money can an individual take offshore in any year?"}]
        testing = True
        manual_responses_for_testing = []
        manual_responses_for_testing.append("Test to see what happens when if the API cannot listen to instructions")
        manual_responses_for_testing.append("SECTION: but after marking its own homework it behaves")

        flag, response = self.excon_test.resource_augmented_query(model_to_use="gpt-3.5-turbo", 
                                                                temperature = 0, 
                                                                max_tokens = 200, 
                                                                df_definitions = relevant_definitions, 
                                                                df_search_sections = relevant_sections,
                                                                testing = testing,
                                                                manual_responses_for_testing = manual_responses_for_testing)
        assert flag == self.excon_test.rag_prefixes[1] # "SECTION:"
        # Also check that nothing happened to the internal message stack
        assert len(self.excon_test.messages) == 1

        # Check the stubbornly disobedient branch
        self.excon_test.messages = [{"role": "user", "content": "How much money can an individual take offshore in any year?"}]
        testing = True
        manual_responses_for_testing = []
        manual_responses_for_testing.append("Test to see what happens when if the API cannot listen to instructions")
        manual_responses_for_testing.append("But even after marking its own homework it cannot listen to instructions")

        flag, response = self.excon_test.resource_augmented_query(model_to_use="gpt-3.5-turbo", 
                                                                temperature = 0, 
                                                                max_tokens = 200, 
                                                                df_definitions = relevant_definitions, 
                                                                df_search_sections = relevant_sections,
                                                                testing = testing,
                                                                manual_responses_for_testing = manual_responses_for_testing)
        assert flag == self.excon_test.rag_prefixes[3] # FAIL:
        # Also check that nothing happened to the internal message stack
        assert len(self.excon_test.messages) == 1

        if self.include_calls_to_api:
            # Manually force the first API response to get to the second loop, then test the second API call
            self.excon_test.system_state = "rag"
            self.excon_test.messages = [{"role": "user", "content": user_context}]
            # NOTE: I am not going to test the openai api call. I am going to use 'testing' mode with canned answers
            testing = True
            manual_responses_for_testing = []
            manual_responses_for_testing.append("The text does not help answer the question")
            flag, response = self.excon_test.resource_augmented_query(model_to_use="gpt-3.5-turbo", 
                                                                    temperature = 0, 
                                                                    max_tokens = 200, 
                                                                    df_definitions = relevant_definitions, 
                                                                    df_search_sections = relevant_sections,
                                                                    testing = testing,
                                                                    manual_responses_for_testing = manual_responses_for_testing)
            assert flag == self.excon_test.rag_prefixes[0] # "ANSWER:"
            # Also check that nothing happened to the internal message stack
            assert len(self.excon_test.messages) == 1


    def test_similarity_search(self):
        if self.include_calls_to_api:
            # Check that random chit-chat to the main dataset does not return any hits from the embeddings
            text = "Hi"
            df_definitions, df_search_sections = self.excon.similarity_search(text)
            assert len(df_definitions) == 0
            assert len(df_search_sections) == 0 
            # now move to the testing dataset for fine grained tests
            user_context = "Who can trade gold?"
            relevant_definitions, relevant_sections = self.excon_test.similarity_search(user_context)
            assert len(relevant_definitions) == 0
            assert len(relevant_sections) == 2
            assert relevant_sections.iloc[0]["reference"] == 'C.(C)'
            assert relevant_sections.iloc[1]["reference"] == 'C.(G)'


    def test__filter_relevant_sections(self):
        test_data = []
        df_test_data = pd.DataFrame(test_data, columns = ["section", "cosine_distance"])        
        df_filtered_test_data = self.excon._filter_relevant_sections(df_test_data)
        assert len(df_filtered_test_data) == 0
        
        test_data.append(['A.1(A)(i)(a)', 0.1])
        df_test_data = pd.DataFrame(test_data, columns = ["section", "cosine_distance"])        
        df_filtered_test_data = self.excon._filter_relevant_sections(df_test_data)
        assert len(df_filtered_test_data) == 1
        assert df_filtered_test_data.iloc[0]["count"] == 1
        # Check that duplicate top values are filtered and the cosine distance is the minimum
        test_data.append(['A.1(A)(i)(a)', 0.2])
        df_test_data = pd.DataFrame(test_data, columns = ["section", "cosine_distance"])        
        df_filtered_test_data = self.excon._filter_relevant_sections(df_test_data)
        assert len(df_filtered_test_data) == 1
        assert df_filtered_test_data.iloc[0]["cosine_distance"] == 0.1
        # Check that the top and mode results are returned and the mode value is the lowest distance
        test_data = []
        test_data.append(['A.1(A)(i)(a)', 0.1])
        test_data.append(['A.1(A)(i)(b)', 0.2])
        test_data.append(['A.1(A)(i)(b)', 0.3])
        df_test_data = pd.DataFrame(test_data, columns = ["section", "cosine_distance"])        
        df_filtered_test_data = self.excon._filter_relevant_sections(df_test_data)
        assert len(df_filtered_test_data) == 2
        assert df_filtered_test_data.iloc[1]["cosine_distance"] == 0.2 # Note the order of the search_section is preserved so the mode should be second
        assert df_filtered_test_data.iloc[1]["count"] == 2

        # Check if there are no duplicate indexes that we still return multiple sections
        test_data = []
        test_data.append(['A.1(A)(i)(a)', 0.1])
        test_data.append(['A.1(A)(i)(b)', 0.2])
        df_test_data = pd.DataFrame(test_data, columns = ["section", "cosine_distance"])        
        df_filtered_test_data = self.excon._filter_relevant_sections(df_test_data)
        assert len(df_filtered_test_data) == 2
        assert df_filtered_test_data.iloc[0]["reference"] == 'A.1(A)(i)(a)'
        assert df_filtered_test_data.iloc[0]["cosine_distance"] == 0.1
        assert df_filtered_test_data.iloc[1]["reference"] == 'A.1(A)(i)(b)'
        assert df_filtered_test_data.iloc[1]["cosine_distance"] == 0.2
        test_data = []
        test_data.append(['A.1(A)(i)(a)', 0.1])
        test_data.append(['A.1(A)(i)(b)', 0.2])
        test_data.append(['A.1(A)(i)(c)', 0.3])
        df_test_data = pd.DataFrame(test_data, columns = ["section", "cosine_distance"])        
        df_filtered_test_data = self.excon._filter_relevant_sections(df_test_data)
        assert len(df_filtered_test_data) == 3
        assert df_filtered_test_data.iloc[0]["reference"] == 'A.1(A)(i)(a)'
        assert df_filtered_test_data.iloc[0]["cosine_distance"] == 0.1
        assert df_filtered_test_data.iloc[1]["reference"] == 'A.1(A)(i)(b)'
        assert df_filtered_test_data.iloc[1]["cosine_distance"] == 0.2
        assert df_filtered_test_data.iloc[2]["reference"] == 'A.1(A)(i)(c)'
        assert df_filtered_test_data.iloc[2]["cosine_distance"] == 0.3
        test_data = []
        test_data.append(['A.1(A)(i)(a)', 0.1])
        test_data.append(['A.1(A)(i)(b)', 0.2])
        test_data.append(['A.1(A)(i)(c)', 0.3])
        test_data.append(['A.1(A)(i)(d)', 0.3])
        df_test_data = pd.DataFrame(test_data, columns = ["section", "cosine_distance"])        
        df_filtered_test_data = self.excon._filter_relevant_sections(df_test_data)
        assert len(df_filtered_test_data) == 3
        assert df_filtered_test_data.iloc[0]["reference"] == 'A.1(A)(i)(a)'
        assert df_filtered_test_data.iloc[0]["cosine_distance"] == 0.1
        assert df_filtered_test_data.iloc[1]["reference"] == 'A.1(A)(i)(b)'
        assert df_filtered_test_data.iloc[1]["cosine_distance"] == 0.2
        assert df_filtered_test_data.iloc[2]["reference"] == 'A.1(A)(i)(c)'
        assert df_filtered_test_data.iloc[2]["cosine_distance"] == 0.3


        # My logic should exclude the second most likely search item
        test_data = []
        test_data.append(['A.1(A)(i)(a)', 0.1])
        test_data.append(['A.1(A)(i)(b)', 0.2])
        test_data.append(['A.1(A)(i)(c)', 0.3])
        test_data.append(['A.1(A)(i)(c)', 0.4])
        df_test_data = pd.DataFrame(test_data, columns = ["section", "cosine_distance"])        
        df_filtered_test_data = self.excon._filter_relevant_sections(df_test_data)
        assert len(df_filtered_test_data) == 2
        assert df_filtered_test_data.iloc[1]["cosine_distance"] == 0.3
        # mode plus replete
        test_data = []
        test_data.append(['A.1(A)(i)(a)', 0.1])
        test_data.append(['A.1(A)(i)(b)', 0.2])
        test_data.append(['A.1(A)(i)(c)', 0.3])
        test_data.append(['A.1(A)(i)(c)', 0.4])
        test_data.append(['A.1(A)(i)(c)', 0.5])
        test_data.append(['A.1(A)(i)(b)', 0.6])
        df_test_data = pd.DataFrame(test_data, columns = ["section", "cosine_distance"])        
        df_filtered_test_data = self.excon._filter_relevant_sections(df_test_data)
        assert len(df_filtered_test_data) == 3
        assert df_filtered_test_data.iloc[1]["reference"] == "A.1(A)(i)(c)"
        assert df_filtered_test_data.iloc[1]["cosine_distance"] == 0.3
        assert df_filtered_test_data.iloc[1]["count"] == 3
        assert df_filtered_test_data.iloc[2]["reference"] == "A.1(A)(i)(b)"
        assert df_filtered_test_data.iloc[2]["cosine_distance"] == 0.2
        assert df_filtered_test_data.iloc[2]["count"] == 2

        # mode plus two repletes
        test_data = []
        test_data.append(['A.1(A)(i)(a)', 0.1])
        test_data.append(['A.1(A)(i)(b)', 0.2])
        test_data.append(['A.1(A)(i)(c)', 0.3])
        test_data.append(['A.1(A)(i)(d)', 0.35])
        test_data.append(['A.1(A)(i)(e)', 0.375])
        test_data.append(['A.1(A)(i)(c)', 0.4])
        test_data.append(['A.1(A)(i)(c)', 0.5])
        test_data.append(['A.1(A)(i)(b)', 0.6])
        test_data.append(['A.1(A)(i)(d)', 0.7])
        df_test_data = pd.DataFrame(test_data, columns = ["section", "cosine_distance"])        
        df_filtered_test_data = self.excon._filter_relevant_sections(df_test_data)
        assert len(df_filtered_test_data) == 4
        # Note the order of the search_section is preserved so the mode should be second
        assert df_filtered_test_data.iloc[3]["reference"] == "A.1(A)(i)(d)"
        assert df_filtered_test_data.iloc[3]["cosine_distance"] == 0.35
        # no unique mode
        test_data = []
        test_data.append(['A.1(A)(i)(a)', 0.1])
        test_data.append(['A.1(A)(i)(b)', 0.2])
        test_data.append(['A.1(A)(i)(c)', 0.3])
        test_data.append(['A.1(A)(i)(c)', 0.5])
        test_data.append(['A.1(A)(i)(b)', 0.6])
        test_data.append(['A.1(A)(i)(d)', 0.7])
        df_test_data = pd.DataFrame(test_data, columns = ["section", "cosine_distance"])        
        df_filtered_test_data = self.excon._filter_relevant_sections(df_test_data)
        assert len(df_filtered_test_data) == 3


    def test_get_regulation_detail(self):
        section_reference = 'B.18(B)(i)(b)'
        expected_text = "B.18 Control of exports - general\n\
    (B) Regulations in respect of goods exported for sale abroad\n\
        (i) Authorised Dealers must ensure that all exporters are aware of their legal obligation in terms of the provisions of Regulations 6, 10 and 11 to:\n\
            (b) receive the full foreign currency proceeds not later than six months from the date of shipment. Authorised Dealers may authorise South African exporters to grant credit of up to 12 months to foreign importers, provided that the Authorised Dealer granting the authority is satisfied that the credit is necessary in the particular trade or that it is needed to protect an existing export market or to capture a new export market. In this regard, Authorised Dealers are requested to specifically draw the attention of exporters to the provisions of Regulation 6(1) and (5);"

        retrieved_text = self.excon.get_regulation_detail(section_reference)
        assert retrieved_text == expected_text
        
        # A problem case that should now be fixed
        retrieved_text = self.excon.get_regulation_detail('B.10(C)(i)(c)')
        expected_text = "B.10 Insurance and pensions\n    (C) Foreign currency payments in respect of short-term insurance premiums or reinsurance premiums\n        (i) In respect of insurance and reinsurance premiums placed abroad, Authorised Dealers may approve the following:\n            (c) Insurance (excluding reinsurance) through Lloyd's correspondents approved by Lloyd's of London\n            Applications by Lloyd's correspondents approved by Lloyd's of London to remit insurance premiums, excluding insurance premiums in respect of currency risks, in respect of:\n                (aa) cover placed in its entirety with Lloyd's underwriters through a broker at Lloyd's, which request must be accompanied by a letter signed by two senior officials of the Lloyd's correspondent concerned incorporating:\n                    (1) a declaration that the Lloyd's correspondent is authorised to carry on such insurance business under the Short-term Insurance Act; and\n                    (2) a declaration that the transaction was entered into with an underwriter at Lloyd's through a broker at Lloyd's.\n                (bb) cover placed through a broker at Lloyd's which is not in its entirety underwritten by an underwriter at Lloyd's which request must be accompanied by:\n                    (1) a letter signed by two senior officials of the Lloyd's correspondent declaring that the Lloyd's correspondent is authorised to carry on such insurance business under the Short-term Insurance Act; and\n                    (2) a copy of a letter  issued by the Registrar of Short-term Insurance, granting approval  in terms of section 8(2)(d) of the Short-term Insurance Act to the intermediary/ Lloyd's correspondent to render services in relation to that short-term policy."
        assert retrieved_text == expected_text

    def test__find_reference_that_calls_for(self):
        text = "I.2 Local facilities to non-residents\n\
                    (cc) The overall finance period, including any initial credit granted by the exporter, may not exceed six months from date of shipment of the underlying goods from South Africa unless the dispensation outlined in section B.18(B)(i)(b) of the Authorised Dealer Manual has been granted, when the overall finance period, including any initial credit granted by the exporter, may not exceed 12 months from date of shipment of the underlying goods from South Africa. An export finance facility may be extended in the event of the overseas importer requiring an extension of the original credit period, provided that the overall finance periods set out above are not exceeded."
        manual_data = []
        manual_data.append(["I.2(A)(i)(a)(cc)", 1, text, 100])
        df_manual_data = pd.DataFrame(manual_data, columns = ["reference", "count", "raw_text", "token_count"])
        referring_sections = self.excon._find_reference_that_calls_for("B.18(B)(i)(b)", df_manual_data)
        assert len(referring_sections) == 1
        assert referring_sections[0] == "I.2(A)(i)(a)(cc)"

        # Add a second reference to the RAG data
        text_2 = "I.2 Local facilities to non-residents\n\
                    (dd) Random Text with no reference"
        manual_data.append(["I.2(A)(i)(a)(dd)", 1, text_2, 100])
        df_manual_data = pd.DataFrame(manual_data, columns = ["reference", "count", "raw_text", "token_count"])
        referring_sections = self.excon._find_reference_that_calls_for("B.18(B)(i)(b)", df_manual_data)
        assert len(referring_sections) == 1
        assert referring_sections[0] == "I.2(A)(i)(a)(cc)"

        # Add a third reference to the RAG data
        text_3 = "I.2 Local facilities to non-residents\n\
                    (ee) Random Text with another reference to B.18(B) (i) (b) but with some random spaces"
        manual_data.append(["I.2(A)(i)(a)(ee)", 1, text_3, 100])
        df_manual_data = pd.DataFrame(manual_data, columns = ["reference", "count", "raw_text", "token_count"])
        referring_sections = self.excon._find_reference_that_calls_for("B.18(B)(i)(b)", df_manual_data)
        assert len(referring_sections) == 2
        assert referring_sections[0] == "I.2(A)(i)(a)(cc)"
        assert referring_sections[1] == "I.2(A)(i)(a)(ee)"


    def test_add_section_to_resource(self):
        # Note that I need to use references that appear in the test data
        text = "A.3 Duties and responsibilities of Authorised Dealers\n\
                    (A) Introduction\n\
                        (i) Fake reference to B.4(B)(iv)(f)"
        manual_data = []
        manual_data.append(["A.3(A)(i)", 1, text, 100])
        df_manual_data = pd.DataFrame(manual_data, columns = ["reference", "count", "raw_text", "token_count"])
        df_updated = self.excon_test.add_section_to_resource('B.4(B)(iv)(f)', df_manual_data)
        assert len(df_updated) == 2
        assert df_updated.iloc[0]['reference'] == "A.3(A)(i)"
        assert df_updated.iloc[1]['reference'] == 'B.4(B)(iv)(f)'

        # Add a second reference to the RAG data
        text_2 = "A.3 Duties and responsibilities of Authorised Dealers\n\
                    (A) Introduction\n\
                        (ii) No references to be found here"
        manual_data.append(["A.3(A)(ii)", 1, text_2, 100])
        df_manual_data = pd.DataFrame(manual_data, columns = ["reference", "count", "raw_text", "token_count"])
        df_updated = self.excon_test.add_section_to_resource('B.4(B)(iv)(f)', df_manual_data)
        assert len(df_updated) == 2
        assert df_updated.iloc[0]['reference'] == "A.3(A)(i)"
        assert df_updated.iloc[1]['reference'] == 'B.4(B)(iv)(f)'

        # Add a third reference to the RAG data
        text_3 = "A.3 Local facilities to non-residents\n\
                    (B) Random Text with another reference to B.4(B) (iv) (f) but with some random spaces"
        manual_data.append(["A.3(B)", 1, text_3, 100])
        df_manual_data = pd.DataFrame(manual_data, columns = ["reference", "count", "raw_text", "token_count"])
        df_updated = self.excon_test.add_section_to_resource('B.4(B)(iv)(f)', df_manual_data)
        assert len(df_updated) == 3
        assert df_updated.iloc[0]['reference'] == "A.3(A)(i)"
        assert df_updated.iloc[1]['reference'] == "A.3(B)"
        assert df_updated.iloc[2]['reference'] == 'B.4(B)(iv)(f)'


    def test__find_fuzzy_reference(self):
        text = "(cc) unless the dispensation outlined in section B.18(B) (i) (b) of the Authorised Dealer Manual has been granted"        
        section = "B.18(B)(i)(b)"
        match = self.excon._find_fuzzy_reference(text, section)
        assert match is not None

        