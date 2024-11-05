from time import sleep
class Moderator:
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
    def __call__(self, query: str):
        
        SLEEP_TIME = 0.5
        MAX_TRIES = 10
        
        realtime_reqs_instruction = "If the above query has real time requirements (or) contains key words like\n\
        'latest', 'current', 'today', 'trending'...etc then reply with yes with 'yes' otherwise reply with no"
        ambiguity_instruction = "If the above query can be interpreted in multiple ways or if it can have multiple answers then reply with 'yes' otherwise\n\
        reply with 'no'"
        
        realtime_reqs_examples = "\
        Q: who is the current prime minister of India\n\
        A: yes\n\
        Q: how is the weather tonight in khammam?\n\
        A: yes\n\
        Q: who was president of USA during 2010?\n\
        A: no\n\
        Q: what is an llm?\n\
        A: no"
        
        ambiguity_examples = "\
        Q: who is the prime minister?\n\
        A: yes\n\
        Q: when is the tournament held?\n\
        A: yes\n\
        Q: who won 2021 IPL finals?\n\
        A: no\n\
        Q: Harry potter was written by?\n\
        A: no"
        for i in range(MAX_TRIES):
            try:
                contains_realtime_reqs = llm.invoke(f"query: {query} \n\n instruction\
                : {realtime_reqs_instruction}\n\n examples: \n {realtime_reqs_examples}").content.strip().lower()
                break
            except:
                if i == MAX_TRIES-1:
                    raise Exception("Max tries exceeded")
                continue
        assert contains_realtime_reqs == 'yes' or contains_realtime_reqs == 'no'
        if contains_realtime_reqs == 'yes':
            return 'websearch' # later in the pipeline analyze the websearch results and try RAG only if the results found are unsatisfactory
        elif contains_realtime_reqs == 'no':
            for i in range(MAX_TRIES):
                try:
                    contains_ambiguity = llm.invoke(f"query: {query} \n\n instruction\
                    : {ambiguity_instruction}\n\n examples: \n {ambiguity_examples}").content.strip().lower()
                    break
                except:
                    if i == MAX_TRIES-1:
                        raise Exception("Max tries exceeded")
                    continue
            assert contains_ambiguity == 'yes' or contains_ambiguity == 'no'
            if contains_ambiguity == 'yes':
                return 'rag'
            answer = llm.invoke(query).content
            answer_satisfactory_instruction = "if you find the answer token to properly answer the query token then respond with 'yes' otherwise 'no'"
            for i in range(MAX_TRIES):
                try:
                    answer_satisfactory = llm.invoke(f"query: {query}\n\n answer: {answer}\n\n instruction: {answer_satisfactory_instruction}").content.strip().lower()
                    break
                except:
                    if i == MAX_TRIES-1:
                        raise Exception("Max tries exceeded")
                    continue
            assert answer_satisfactory == 'yes' or answer_satisfactory == 'no'
                        
            return 'llm' if answer_satisfactory == 'yes' else 'rag'
