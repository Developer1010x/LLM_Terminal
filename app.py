#!/usr/bin/env python3
"""
LLM-Enhanced Cross-Platform Terminal
A powerful terminal that combines features from Bash, Zsh, and PowerShell
with LLM integration for intelligent command resolution and error handling.
"""

import os
import sys
import subprocess
import platform
import json
import re
import shlex
import asyncio
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from collections import defaultdict, Counter

# Import local LLM module
try:
    from llm import ask_llm, get_model_name
    HAS_LOCAL_LLM = True
except ImportError:
    HAS_LOCAL_LLM = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter, PathCompleter
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.shortcuts import print_formatted_text
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False

@dataclass
class CommandResult:
    """Result of command execution"""
    success: bool
    output: str
    error: str
    exit_code: int
    execution_time: float

@dataclass
class HistoryEntry:
    """Enhanced history entry with metadata"""
    command: str
    timestamp: str
    directory: str
    success: bool
    execution_time: float
    line_number: int
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)

@dataclass
class ShellConfig:
    """Configuration for the shell"""
    llm_enabled: bool = True
    auto_suggest: bool = True
    smart_completion: bool = True
    cross_platform_aliases: bool = True
    history_file: str = "~/.llm_shell_history"
    enhanced_history_file: str = "~/.llm_shell_enhanced_history.json"
    config_file: str = "~/.llm_shell_config.json"
    max_history_entries: int = 10000
    show_recommendations: bool = True

class HistoryManager:
    """Advanced history management with analytics and recommendations"""
    
    def __init__(self, history_file: str, max_entries: int = 10000):
        self.history_file = os.path.expanduser(history_file)
        self.max_entries = max_entries
        self.entries: List[HistoryEntry] = []
        self.command_frequency = Counter()
        self.directory_frequency = defaultdict(Counter)
        self.load_history()
    
    def load_history(self):
        """Load history from file"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.entries = [HistoryEntry.from_dict(entry) for entry in data]
                    
                # Rebuild frequency counters
                for entry in self.entries:
                    self.command_frequency[entry.command] += 1
                    self.directory_frequency[entry.directory][entry.command] += 1
                    
        except Exception as e:
            print(f"Warning: Could not load history: {e}")
            self.entries = []
    
    def save_history(self):
        """Save history to file"""
        try:
            # Limit history size
            if len(self.entries) > self.max_entries:
                self.entries = self.entries[-self.max_entries:]
            
            with open(self.history_file, 'w') as f:
                json.dump([entry.to_dict() for entry in self.entries], f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save history: {e}")
    
    def add_entry(self, command: str, directory: str, success: bool, execution_time: float):
        """Add new command to history"""
        entry = HistoryEntry(
            command=command,
            timestamp=datetime.now().isoformat(),
            directory=directory,
            success=success,
            execution_time=execution_time,
            line_number=len(self.entries) + 1
        )
        
        self.entries.append(entry)
        self.command_frequency[command] += 1
        self.directory_frequency[directory][command] += 1
        self.save_history()
    
    def find_previous_usage(self, command: str) -> Optional[HistoryEntry]:
        """Find the first usage of a command"""
        for entry in self.entries:
            if entry.command == command:
                return entry
        return None
    
    def get_command_recommendations(self, current_dir: str, limit: int = 5) -> List[Tuple[str, int, str]]:
        """Get command recommendations based on frequency and context"""
        recommendations = []
        
        # Context-aware recommendations (commands used in current directory)
        dir_commands = self.directory_frequency.get(current_dir, Counter())
        for cmd, count in dir_commands.most_common(limit):
            recommendations.append((cmd, count, "directory context"))
        
        # Global frequent commands
        global_commands = []
        for cmd, count in self.command_frequency.most_common(limit * 2):
            if cmd not in [r[0] for r in recommendations]:
                global_commands.append((cmd, count, "global frequency"))
        
        recommendations.extend(global_commands[:limit - len(recommendations)])
        return recommendations[:limit]
    
    def get_similar_commands(self, partial_command: str, limit: int = 3) -> List[str]:
        """Get commands that start with the partial command"""
        matches = []
        for cmd in self.command_frequency.keys():
            if cmd.startswith(partial_command) and cmd != partial_command:
                matches.append(cmd)
        
        # Sort by frequency
        matches.sort(key=lambda x: self.command_frequency[x], reverse=True)
        return matches[:limit]
    
    def get_history_stats(self) -> Dict[str, Any]:
        """Get statistics about command history"""
        if not self.entries:
            return {"total_commands": 0}
        
        successful_commands = sum(1 for entry in self.entries if entry.success)
        avg_execution_time = sum(entry.execution_time for entry in self.entries) / len(self.entries)
        
        # Most used directories
        dir_usage = Counter(entry.directory for entry in self.entries)
        
        # Recent activity (last 10 commands)
        recent_commands = [entry.command for entry in self.entries[-10:]]
        
        return {
            "total_commands": len(self.entries),
            "successful_commands": successful_commands,
            "success_rate": f"{(successful_commands / len(self.entries) * 100):.1f}%",
            "avg_execution_time": f"{avg_execution_time:.3f}s",
            "most_used_commands": self.command_frequency.most_common(5),
            "most_used_directories": dir_usage.most_common(3),
            "recent_commands": recent_commands
        }

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def complete(self, prompt: str) -> str:
        pass
    
    @abstractmethod
    async def analyze_error(self, command: str, error: str, context: str) -> str:
        pass

class OllamaProvider(LLMProvider):
    """Ollama provider using local LLM setup"""
    
    def __init__(self):
        if not HAS_LOCAL_LLM:
            raise ImportError("Local LLM module not found. Make sure llm.py is in the same directory.")
        
        self.model = get_model_name()
        print(f"✓ Using Ollama model: {self.model}")
    
    async def complete(self, prompt: str) -> str:
        """Generate command completion suggestions"""
        system_context = """You are a helpful terminal assistant. Given a partial command or description, 
suggest the most appropriate shell command. Be concise and provide only the command, no explanations unless asked.

Examples:
- "list files" -> "ls -la"
- "find python files" -> "find . -name '*.py'"
- "check running processes" -> "ps aux"
"""
        
        full_prompt = f"{system_context}\n\nUser request: {prompt}\n\nSuggested command:"
        
        try:
            response = ask_llm(full_prompt)
            # Clean up the response to extract just the command
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('//'):
                    return line
            return response.strip()
        except Exception as e:
            return f"LLM Error: {str(e)}"
    
    async def analyze_error(self, command: str, error: str, context: str) -> str:
        """Analyze command errors and suggest fixes"""
        system_context = """You are a terminal debugging assistant. Analyze the failed command and error message, 
then provide a clear explanation of what went wrong and suggest a corrected command. Be concise but helpful.

Format your response as:
Problem: [brief explanation]
Solution: [corrected command or fix]
"""
        
        full_prompt = f"""{system_context}

Failed Command: {command}
Error Message: {error}
Context: {context}

Analysis:"""
        
        try:
            response = ask_llm(full_prompt)
            return response.strip()
        except Exception as e:
            return f"LLM Error: {str(e)}"

class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider for LLM functionality (kept for compatibility)"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        if not HAS_OPENAI:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")
        
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
    
    async def complete(self, prompt: str) -> str:
        """Generate command completion suggestions"""
        system_prompt = """You are a helpful terminal assistant. Given a partial command or description, 
        suggest the most appropriate shell command. Be concise and provide only the command, no explanations unless asked."""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"LLM Error: {str(e)}"
    
    async def analyze_error(self, command: str, error: str, context: str) -> str:
        """Analyze command errors and suggest fixes"""
        system_prompt = """You are a terminal debugging assistant. Analyze the failed command and error message, 
        then provide a clear explanation of what went wrong and suggest a corrected command. Be concise but helpful."""
        
        user_prompt = f"""
        Command: {command}
        Error: {error}
        Context: {context}
        OS: {platform.system()}
        """
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"LLM Error: {str(e)}"

class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing without API keys"""
    
    async def complete(self, prompt: str) -> str:
        # Simple command suggestions based on keywords
        suggestions = {
            "list": "ls -la" if platform.system() != "Windows" else "dir",
            "find": "find . -name",
            "search": "grep -r",
            "copy": "cp" if platform.system() != "Windows" else "copy",
            "move": "mv" if platform.system() != "Windows" else "move",
            "delete": "rm" if platform.system() != "Windows" else "del",
            "process": "ps aux" if platform.system() != "Windows" else "tasklist",
        }
        
        for keyword, suggestion in suggestions.items():
            if keyword in prompt.lower():
                return suggestion
        
        return "# No suggestion available - using mock LLM provider"
    
    async def analyze_error(self, command: str, error: str, context: str) -> str:
        common_fixes = {
            "permission denied": "Try running with elevated privileges (sudo/admin)",
            "command not found": "Check if the command is installed and in your PATH",
            "no such file": "Verify the file path exists",
            "access denied": "Check file permissions",
        }
        
        error_lower = error.lower()
        for issue, fix in common_fixes.items():
            if issue in error_lower:
                return f"Issue: {issue.title()}\nSuggestion: {fix}"
        
        return "Unable to analyze error - using mock LLM provider"

class CrossPlatformShell:
    """Main shell class with cross-platform functionality"""
    
    def __init__(self, config: ShellConfig):
        self.config = config
        self.current_dir = Path.cwd()
        self.history = []
        self.aliases = {}
        self.environment = dict(os.environ)
        self.llm_provider: Optional[LLMProvider] = None
        self.history_manager = HistoryManager(config.enhanced_history_file, config.max_history_entries)
        
        # Initialize LLM provider
        self._setup_llm()
        
        # Setup cross-platform aliases
        if config.cross_platform_aliases:
            self._setup_cross_platform_aliases()
        
        # Setup prompt toolkit if available
        if HAS_PROMPT_TOOLKIT:
            self.session = PromptSession(
                history=FileHistory(os.path.expanduser(config.history_file)),
                completer=self._create_completer(),
            )
    
    def _setup_llm(self):
        """Initialize LLM provider"""
        try:
            # Priority 1: Try local Ollama setup
            if HAS_LOCAL_LLM:
                # Test if Ollama is actually running
                try:
                    import requests
                    response = requests.get("http://localhost:11434/api/tags", timeout=2)
                    if response.status_code == 200:
                        self.llm_provider = OllamaProvider()
                        return
                    else:
                        print("⚠ Ollama server not responding")
                except Exception as e:
                    print(f"⚠ Cannot connect to Ollama: {e}")
            
            # Priority 2: Try OpenAI if API key is available
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key and HAS_OPENAI:
                self.llm_provider = OpenAIProvider(api_key)
                print("✓ LLM provider initialized (OpenAI)")
                return
            
            # Fallback: Mock provider
            self.llm_provider = MockLLMProvider()
            print("⚠ Using mock LLM provider")
            if not HAS_LOCAL_LLM:
                print("  • llm.py not found - create it in the same directory")
            print("  • For Ollama: Make sure Ollama is running (ollama serve)")
            print("  • For OpenAI: Set OPENAI_API_KEY environment variable")
            
        except Exception as e:
            self.llm_provider = MockLLMProvider()
            print(f"⚠ Falling back to mock LLM provider: {e}")
    
    def _setup_cross_platform_aliases(self):
        """Setup aliases that work across platforms"""
        system = platform.system().lower()
        
        if system == "windows":
            self.aliases.update({
                "ls": "dir",
                "cat": "type",
                "grep": "findstr",
                "ps": "tasklist",
                "kill": "taskkill /F /PID",
                "which": "where",
                "clear": "cls",
                "cp": "copy",
                "mv": "move",
                "rm": "del",
                "pwd": "cd",
            })
        else:  # Unix-like systems
            self.aliases.update({
                "dir": "ls -la",
                "type": "cat",
                "findstr": "grep",
                "tasklist": "ps aux",
                "cls": "clear",
                "copy": "cp",
                "move": "mv",
                "del": "rm",
            })
    
    def _create_completer(self):
        """Create command completer"""
        if not HAS_PROMPT_TOOLKIT:
            return None
        
        # Basic commands for all platforms
        commands = [
            "cd", "ls", "cat", "grep", "find", "ps", "kill", "clear", "exit", "help",
            "llm", "explain", "fix", "suggest", "history", "stats", "search", "recommendations"
        ]
        
        # Add platform-specific commands
        if platform.system() == "Windows":
            commands.extend(["dir", "type", "findstr", "tasklist", "cls"])
        else:
            commands.extend(["sudo", "chmod", "chown", "df", "du", "top"])
        
        return WordCompleter(commands + list(self.aliases.keys()))
    
    def get_prompt(self) -> str:
        """Generate shell prompt"""
        user = os.getenv("USER", os.getenv("USERNAME", "user"))
        hostname = platform.node().split('.')[0]  # Get short hostname
        cwd = str(self.current_dir).replace(str(Path.home()), "~")
        
        # Simple prompt without colors for better compatibility
        return f"{user}@{hostname}:{cwd}$ "
    
    async def execute_command(self, command: str) -> CommandResult:
        """Execute a command and return result"""
        if not command.strip():
            return CommandResult(True, "", "", 0, 0.0)
        
        # Handle built-in commands
        if command.startswith("cd "):
            return self._handle_cd(command)
        elif command in ["exit", "quit"]:
            return CommandResult(True, "Goodbye!", "", 0, 0.0)
        elif command.startswith("llm "):
            return await self._handle_llm_command(command)
        elif command.startswith("explain "):
            return await self._handle_explain_command(command)
        elif command.startswith("fix "):
            return await self._handle_fix_command(command)
        elif command == "history":
            return self._handle_history_command()
        elif command == "stats":
            return self._handle_stats_command()
        elif command.startswith("search "):
            return self._handle_search_command(command)
        elif command in ["rec", "recommendations"]:
            return self._handle_recommendations_command()
        elif command == "help":
            return self._handle_help()
        
        # Check for command repetition and show previous usage
        previous_usage = self.history_manager.find_previous_usage(command)
        if previous_usage and self.config.show_recommendations:
            print(f"📝 This command was previously used at line {previous_usage.line_number} on {previous_usage.timestamp[:19]} in {previous_usage.directory}")
        
        # Apply aliases
        if command.split()[0] in self.aliases:
            cmd_parts = command.split()
            cmd_parts[0] = self.aliases[cmd_parts[0]]
            command = " ".join(cmd_parts)
        
        # Execute external command
        start_time = time.time()
        
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    command, shell=True, capture_output=True, text=True,
                    cwd=str(self.current_dir), env=self.environment
                )
            else:
                result = subprocess.run(
                    command, shell=True, capture_output=True, text=True,
                    cwd=str(self.current_dir), env=self.environment
                )
            
            execution_time = time.time() - start_time
            
            # Add to history
            self.history_manager.add_entry(
                command=command,
                directory=str(self.current_dir),
                success=result.returncode == 0,
                execution_time=execution_time
            )
            
            return CommandResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr,
                exit_code=result.returncode,
                execution_time=execution_time
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Add failed command to history
            self.history_manager.add_entry(
                command=command,
                directory=str(self.current_dir),
                success=False,
                execution_time=execution_time
            )
            
            return CommandResult(
                success=False,
                output="",
                error=str(e),
                exit_code=-1,
                execution_time=execution_time
            )
    
    def _handle_cd(self, command: str) -> CommandResult:
        """Handle cd command"""
        try:
            parts = shlex.split(command)
            if len(parts) == 1:
                # cd with no arguments goes to home
                target = Path.home()
            else:
                target = Path(parts[1]).expanduser()
            
            if target.is_dir():
                self.current_dir = target.resolve()
                os.chdir(str(self.current_dir))
                return CommandResult(True, "", "", 0, 0.0)
            else:
                return CommandResult(False, "", f"cd: {target}: No such directory", 1, 0.0)
        
        except Exception as e:
            return CommandResult(False, "", f"cd: {str(e)}", 1, 0.0)
    
    async def _handle_llm_command(self, command: str) -> CommandResult:
        """Handle LLM-based command suggestions"""
        prompt = command[4:].strip()  # Remove "llm "
        
        if not prompt:
            return CommandResult(False, "", "Usage: llm <description>", 1, 0.0)
        
        try:
            suggestion = await self.llm_provider.complete(prompt)
            return CommandResult(True, f"Suggested command: {suggestion}", "", 0, 0.0)
        except Exception as e:
            return CommandResult(False, "", f"LLM error: {str(e)}", 1, 0.0)
    
    async def _handle_explain_command(self, command: str) -> CommandResult:
        """Explain a command using LLM"""
        cmd_to_explain = command[8:].strip()  # Remove "explain "
        
        if not cmd_to_explain:
            return CommandResult(False, "", "Usage: explain <command>", 1, 0.0)
        
        try:
            explanation = await self.llm_provider.complete(f"Explain this command: {cmd_to_explain}")
            return CommandResult(True, explanation, "", 0, 0.0)
        except Exception as e:
            return CommandResult(False, "", f"LLM error: {str(e)}", 1, 0.0)
    
    def _handle_history_command(self) -> CommandResult:
        """Display command history with enhanced information"""
        if not self.history_manager.entries:
            return CommandResult(True, "No command history available", "", 0, 0.0)
        
        # Show last 20 commands with details
        recent_entries = self.history_manager.entries[-20:]
        output_lines = ["Recent Command History:", "=" * 50]
        
        for entry in recent_entries:
            status = "✓" if entry.success else "✗"
            timestamp = entry.timestamp[:19].replace('T', ' ')
            directory = entry.directory.replace(str(Path.home()), "~")
            
            output_lines.append(
                f"{entry.line_number:4d} {status} {timestamp} [{directory}] {entry.command}"
            )
        
        return CommandResult(True, "\n".join(output_lines), "", 0, 0.0)
    
    def _handle_stats_command(self) -> CommandResult:
        """Display command usage statistics"""
        stats = self.history_manager.get_history_stats()
        
        if stats["total_commands"] == 0:
            return CommandResult(True, "No command history available for statistics", "", 0, 0.0)
        
        output_lines = [
            "📊 Command Usage Statistics",
            "=" * 40,
            f"Total Commands: {stats['total_commands']}",
            f"Successful Commands: {stats['successful_commands']}",
            f"Success Rate: {stats['success_rate']}",
            f"Average Execution Time: {stats['avg_execution_time']}",
            "",
            "🔥 Most Used Commands:",
        ]
        
        for cmd, count in stats["most_used_commands"]:
            output_lines.append(f"  {count:3d}× {cmd}")
        
        output_lines.extend([
            "",
            "📁 Most Used Directories:",
        ])
        
        for directory, count in stats["most_used_directories"]:
            dir_short = directory.replace(str(Path.home()), "~")
            output_lines.append(f"  {count:3d}× {dir_short}")
        
        output_lines.extend([
            "",
            "🕒 Recent Commands:",
            "  " + " → ".join(stats["recent_commands"][-5:])
        ])
        
        return CommandResult(True, "\n".join(output_lines), "", 0, 0.0)
    
    def _handle_search_command(self, command: str) -> CommandResult:
        """Search command history"""
        query = command[7:].strip()  # Remove "search "
        
        if not query:
            return CommandResult(False, "", "Usage: search <query>", 1, 0.0)
        
        matches = []
        for entry in self.history_manager.entries:
            if query.lower() in entry.command.lower():
                matches.append(entry)
        
        if not matches:
            return CommandResult(True, f"No commands found matching '{query}'", "", 0, 0.0)
        
        output_lines = [f"🔍 Found {len(matches)} commands matching '{query}':", ""]
        
        for entry in matches[-10:]:  # Show last 10 matches
            status = "✓" if entry.success else "✗"
            timestamp = entry.timestamp[:19].replace('T', ' ')
            directory = entry.directory.replace(str(Path.home()), "~")
            
            output_lines.append(
                f"{entry.line_number:4d} {status} {timestamp} [{directory}] {entry.command}"
            )
        
        return CommandResult(True, "\n".join(output_lines), "", 0, 0.0)
    
    def _handle_recommendations_command(self) -> CommandResult:
        """Show command recommendations based on context and frequency"""
        recommendations = self.history_manager.get_command_recommendations(str(self.current_dir))
        
        if not recommendations:
            return CommandResult(True, "No recommendations available yet. Use more commands to build history!", "", 0, 0.0)
        
        output_lines = [
            "💡 Command Recommendations",
            "=" * 30,
            f"Based on your usage in: {str(self.current_dir).replace(str(Path.home()), '~')}",
            ""
        ]
        
        for i, (cmd, count, reason) in enumerate(recommendations, 1):
            output_lines.append(f"{i}. {cmd}")
            output_lines.append(f"   Used {count} times ({reason})")
            output_lines.append("")
        
        # Add similar commands if user typed something partially
        if hasattr(self, '_last_partial_command'):
            similar = self.history_manager.get_similar_commands(self._last_partial_command)
            if similar:
                output_lines.extend([
                    "🔍 Similar Commands:",
                    *[f"  • {cmd}" for cmd in similar]
                ])
        
        return CommandResult(True, "\n".join(output_lines), "", 0, 0.0)
    
    async def _handle_fix_command(self, command: str) -> CommandResult:
        """Fix the last failed command using LLM"""
        if not self.history_manager.entries:
            return CommandResult(False, "", "No previous command to fix", 1, 0.0)
        
        # Find the last failed command
        last_failed = None
        for entry in reversed(self.history_manager.entries):
            if not entry.success:
                last_failed = entry
                break
        
        if not last_failed:
            return CommandResult(True, "No failed commands found in recent history", "", 0, 0.0)
        
        try:
            context = f"Working directory: {self.current_dir}\nOS: {platform.system()}\nPrevious directory: {last_failed.directory}"
            
            fix_suggestion = await self.llm_provider.analyze_error(
                last_failed.command, "Command failed", context
            )
            
            output_lines = [
                f"🔧 Analyzing failed command from line {last_failed.line_number}:",
                f"Command: {last_failed.command}",
                f"Directory: {last_failed.directory}",
                f"Time: {last_failed.timestamp[:19]}",
                "",
                "💡 AI Suggestion:",
                fix_suggestion
            ]
            
            return CommandResult(True, "\n".join(output_lines), "", 0, 0.0)
        except Exception as e:
            return CommandResult(False, "", f"LLM error: {str(e)}", 1, 0.0)
    
    def _handle_help(self) -> CommandResult:
        """Show help information"""
        help_text = """
LLM-Enhanced Cross-Platform Terminal

Built-in Commands:
  cd <dir>          Change directory
  llm <description> Get command suggestions from LLM
  explain <cmd>     Explain what a command does
  fix               Analyze and fix the last failed command
  history           Show recent command history with details
  stats             Show command usage statistics
  search <query>    Search command history
  rec/recommendations Show command recommendations
  help              Show this help message
  exit/quit         Exit the shell

Features:
  • Cross-platform aliases (ls/dir, cat/type, etc.)
  • LLM-powered command suggestions and error analysis
  • Smart completion and history with analytics
  • Command repetition detection and line number tracking
  • Frequency-based recommendations
  • Enhanced search and statistics
  • Unified shell experience across Windows, macOS, and Linux

Environment:
  OS: {platform.system()}
  Python: {sys.version.split()[0]}
  LLM Provider: {'Ollama (' + get_model_name() + ')' if HAS_LOCAL_LLM else 'OpenAI' if HAS_OPENAI and os.getenv('OPENAI_API_KEY') else 'Mock'}
        """
        
        return CommandResult(True, help_text.strip(), "", 0, 0.0)
    
    async def run(self):
        """Main shell loop"""
        print("🚀 LLM-Enhanced Cross-Platform Terminal")
        print(f"Running on {platform.system()} {platform.release()}")
        print("Type 'help' for available commands or 'exit' to quit.\n")
        
        while True:
            try:
                if HAS_PROMPT_TOOLKIT:
                    command = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: self.session.prompt(self.get_prompt())
                    )
                else:
                    command = input(self.get_prompt())
                
                if command.strip() in ["exit", "quit"]:
                    break
                
                result = await self.execute_command(command)
                
                if result.output:
                    print(result.output)
                
                if result.error and not result.success:
                    print(f"Error: {result.error}")
                    
                    # Auto-suggest fix for failed commands
                    if self.config.llm_enabled and result.exit_code != 0:
                        print("\n💡 Getting AI suggestion for fixing this error...")
                        try:
                            context = f"Working directory: {self.current_dir}\nOS: {platform.system()}"
                            suggestion = await self.llm_provider.analyze_error(
                                command, result.error, context
                            )
                            print(f"🤖 AI Suggestion: {suggestion}")
                        except Exception as e:
                            print(f"Could not get AI suggestion: {e}")
                        print()
                
                # Show recommendations after every few commands
                if (len(self.history_manager.entries) % 5 == 0 and
                    len(self.history_manager.entries) > 0 and
                    self.config.show_recommendations):
                    recommendations = self.history_manager.get_command_recommendations(str(self.current_dir), 3)
                    if recommendations:
                        print("💭 Quick recommendations:", end=" ")
                        rec_commands = [rec[0] for rec in recommendations[:3]]
                        print(" | ".join(rec_commands))
                        print()
            
            except KeyboardInterrupt:
                print("\nUse 'exit' or 'quit' to leave the shell.")
            except EOFError:
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
        
        print("Goodbye! 👋")

def main():
    """Main entry point"""
    config = ShellConfig()
    shell = CrossPlatformShell(config)
    
    try:
        asyncio.run(shell.run())
    except KeyboardInterrupt:
        print("\nShell interrupted.")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
