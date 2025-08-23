from mcp.server.fastmcp import FastMCP

from datetime import datetime
import pytz

mcp = FastMCP("Math")

# Prompts
@mcp.prompt()
def example_prompt(question: str) -> str:
    """Example prompt description"""
    return f"""
    You are a math assistant. Answer the question.
    Question: {question}
    """

@mcp.prompt()
def system_prompt() -> str:
    """System prompt description"""
    return """
    You are an AI assistant use the tools if needed.
    """

# Resources
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"

@mcp.resource("config://app")
def get_config() -> str:
    """Static configuration data"""
    return "App configuration here"

# Tools
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

@mcp.tool()
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current date and time in the specified timezone"""
    try:
        # Get the timezone object
        tz = pytz.timezone(timezone)
        # Get current time in the specified timezone
        current_time = datetime.now(tz)
        # Format the time
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S %Z")
        return formatted_time
    except pytz.exceptions.UnknownTimeZoneError:
        return f"Error: Unknown timezone '{timezone}'. Please use a valid timezone like 'UTC', 'US/Eastern', 'Europe/London', etc."



if __name__ == "__main__":
    mcp.run()  # Run server via stdio