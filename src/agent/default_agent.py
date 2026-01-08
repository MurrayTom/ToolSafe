import re
import json
from copy import deepcopy
from types import FunctionType
from tqdm import tqdm
from agent.agent import Agent_Core

class Default_Agent:
    def __init__(self, system_template="", agentic_model=None, guard_model=None, max_turns=10):
        # Initialize with parent class
        self.system_template= system_template
        self.agentic_model = agentic_model
        self.guard_model = guard_model
        self.max_turns = max_turns
    
    def agent_invoke(self, query="", available_tool_descriptions={}, available_tool_params={}, known_actions={}):
        
        assert isinstance(query, list), "`query` must be a list of messages"

        bot = Agent_Core(agentic_model=self.agentic_model)

        bot.messages = deepcopy(query)

        response = bot.execute()
        bot.messages.append({ 'role': 'assistant', 'content': response })

        return [], bot.messages, ""
