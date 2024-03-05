import subprocess
import datetime


def report_git_status(log_file_path):
    # Define the commands to get the status and diff

    status_command = ["git", "status"]
    diff_command = ["git", "diff"]
    modified_files_command = ["git", "status", "--short"]
    commit_hash_command = ["git", "rev-parse", "HEAD"]

    # Execute the commands and capture their outputs
    try:
        status_result = subprocess.run(status_command, check=True, capture_output=True, text=True)
        diff_result = subprocess.run(diff_command, check=True, capture_output=True, text=True)
        modified_files_result = subprocess.run(modified_files_command, capture_output=True, text=True, check=True)
        commit_hash_result = subprocess.run(commit_hash_command, capture_output=True, text=True, check=True)
        summary = get_git_summary()
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing Git commands: {e}")
        return None, None
    except FileNotFoundError as e:
        print(f"Git is probably not installed or not found in the system PATH: {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None

    # Prepare the content to be written to the log file
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    modified_files_count = len(modified_files_result.stdout.splitlines())
    content = (f"Report Time: {now}\n\n"
               f"Commit: {commit_hash_result.stdout}\n"
               f"Files modified: {modified_files_count}\n"
               f"Summary: {summary}\n\n"
               f"Git Status:\n{status_result.stdout}\n"
               f"Git Diff:\n{diff_result.stdout}")

    # Write the content to the log file
    with open(log_file_path, "a", encoding='utf-8') as log_file:
        log_file.write(content)

    print(f"Git Report saved to {log_file_path}")
    return commit_hash_result.stdout, modified_files_count



def get_git_summary():
    # Get the current branch name
    branch_name = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], capture_output=True, text=True).stdout.strip()

    # Get the commit status relative to the remote branch
    commit_status = subprocess.run(['git', 'rev-list', '--left-right', '--count', 'HEAD...@{u}'], capture_output=True, text=True).stdout.strip().split('\t')
    ahead, behind = commit_status[0], commit_status[1]

    # Get the file status summary
    status_summary = subprocess.run(['git', 'status', '--short'], capture_output=True, text=True).stdout.strip().split('\n')
    added = sum(1 for x in status_summary if x.startswith('A '))
    modified = sum(1 for x in status_summary if x.startswith('M '))
    deleted = sum(1 for x in status_summary if x.startswith('D '))
    untracked = sum(1 for x in status_summary if x.startswith('??'))

    # Construct the summary string
    summary = f"[{branch_name} ↑{ahead} +{added} ~{modified} -{deleted} | ?{untracked}]"

    return summary