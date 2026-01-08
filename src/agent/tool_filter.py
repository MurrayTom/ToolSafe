import re
import json
from copy import deepcopy
from types import FunctionType
from tqdm import tqdm
from agent.agent import *
from agent.agent_prompts import *

from utils.tool_parser import *

class Tool_Filter:
    def __init__(self, system_template="", agentic_model=None, max_turns=10):
        # Initialize with parent class
        self.system_template= system_template
        self.agentic_model = agentic_model
        self.max_turns = max_turns
