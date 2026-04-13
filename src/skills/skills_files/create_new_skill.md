---
name: create_new_skill
description: Use this skill to create a new skill
tools: [write_file, execute_command, compile_skill, read_file]
---

# Instruction to execute an example
1. Use compile_skill tool to create a new skill file with the informations provided by the user
2. Read and edit the new skill file created, with the instructions to execute the skill
2. Read and edit src/skills/myskill.py file properly by adding the new skill class
3. Read and edit src/domains/general.py file properly by adding the new skill class
4. Acknowledge the user that you completed the task and that the agent needs to be restarted to use the new skill
