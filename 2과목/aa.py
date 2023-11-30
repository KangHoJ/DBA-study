import sqlite3
import re
from datetime import datetime

# Connect to SQLite database
conn = sqlite3.connect('minecraft_logs.db')
cursor = conn.cursor()

# Create a table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS minecraft_server_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        log_level TEXT,
        log_message TEXT
    )
''')

# Commit changes and close connection
conn.commit()

# Function to parse and log a single line from the Minecraft server log
def log_to_database(log_line):
    # Example log line: "[00:47:43] [ServerMain/INFO]: Environment: Environment[accountsHost=https://api.mojang.com, ...]"
    log_pattern = r'\[(.*?)\] \[(.*?)\]: (.*)'
    match = re.match(log_pattern, log_line)
    
    if match:
        timestamp_str, log_level, log_message = match.groups()
        timestamp = datetime.strptime(timestamp_str, '%H:%M:%S')
        
        # Insert log into the database
        cursor.execute('INSERT INTO minecraft_server_logs (timestamp, log_level, log_message) VALUES (?, ?, ?)',
                       (timestamp, log_level, log_message))
        conn.commit()

# Assuming you have a list of log lines, you can iterate over them and log to the database
log_lines = [
    "[00:47:43] [ServerMain/INFO]: Environment: Environment[accountsHost=https://api.mojang.com, ...]",
    "[00:47:45] [ServerMain/INFO]: Loaded 7 recipes",
    # ... (more log lines)
]

for log_line in log_lines:
    log_to_database(log_line)

# Close the connection
conn.close()