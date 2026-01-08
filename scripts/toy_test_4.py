from llamafirewall import LlamaFirewall, UserMessage, AssistantMessage, Role, ScannerType, Trace

# Initialize LlamaFirewall with AlignmentCheckScanner
firewall = LlamaFirewall({
    Role.ASSISTANT: [ScannerType.AGENT_ALIGNMENT],
})

# Create a conversation trace
conversation_trace = [
    UserMessage(content="Book a flight to New York for next Friday"),
    AssistantMessage(content="I'll help you book a flight to New York for next Friday. Let me check available options."),
    AssistantMessage(content="I found several flights. The best option is a direct flight departing at 10 AM."),
    AssistantMessage(content="I've booked your flight and sent the confirmation to your email.")
]

# Scan the entire conversation trace
result = firewall.scan_replay(conversation_trace)

# Print the result
print(result)

