import argparse
import os.path
import sys
import termios
import tty
from difflib import unified_diff

from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.prompt import Confirm
from rich.live import Live

from datatrove.io import DataFolder, get_datafolder
from datatrove.pipeline.filters import SamplerFilter
from datatrove.pipeline.readers import CSVReader, JsonlReader, ParquetReader, WarcReader
from datatrove.utils._import_utils import is_rich_available

"""
    Simple utility to compare documents between two datasets by ID.
    Shows side-by-side comparison of documents that have the same ID but different content.
"""

if not is_rich_available():
    raise ImportError("Please install `rich` to run this command (`pip install rich`).")

parser = argparse.ArgumentParser(
    "Compare documents between two datasets by ID. "
    "Shows side-by-side comparison when documents differ. "
    "Any unknown parameters will be passed to the readers (example: 'text_key=text')."
)

parser.add_argument(
    "path1", type=str, help="Path to the first data folder"
)

parser.add_argument(
    "path2", type=str, help="Path to the second data folder"
)

parser.add_argument(
    "-r1",
    "--reader1",
    type=str,
    help="The type of Reader to use for the first dataset. "
    "By default it will be guessed from the file extension. "
    "Can be ('jsonl', 'parquet', 'csv' or 'warc')",
)

parser.add_argument(
    "-r2",
    "--reader2",
    type=str,
    help="The type of Reader to use for the second dataset. "
    "By default it will be guessed from the file extension. "
    "Can be ('jsonl', 'parquet', 'csv' or 'warc')",
)

parser.add_argument(
    "-s", "--sample", type=float, help="Randomly sample a given % of samples from the first dataset. 1.0 to see all samples", default=1.0
)

parser.add_argument(
    "--max-diff-lines", type=int, help="Maximum number of diff lines to show per document", default=20
)

parser.add_argument(
    "--skip-identical", action="store_true", help="Skip documents that are identical between datasets"
)

parser.add_argument(
    "-f", "--filter", type=str, help="Filter the documents by a given expression (ex: len(x.text) > 1000)", default=None
)

console = Console()


def reader_class_from_name(reader_type):
    match reader_type:
        case "jsonl":
            return JsonlReader
        case "csv":
            return CSVReader
        case "parquet":
            return ParquetReader
        case "warc":
            return WarcReader
        case other:
            console.log(f"[red]Unknown reader type {other}")
            sys.exit(-1)


def reader_factory(data_folder: DataFolder, reader_type: str = None, **kwargs):
    """Create appropriate reader for the data folder"""
    data_files = data_folder.list_files()
    if not data_files:
        console.log(f'[red]Could not find any files in "{data_folder.path}"')
        sys.exit(-1)

    if not reader_type:
        match data_files[0][data_files[0].index(".") :]:
            case ".jsonl.gz" | ".jsonl" | ".json":
                reader_type = "jsonl"
            case ".csv":
                reader_type = "csv"
            case ".parquet":
                reader_type = "parquet"
            case ".warc.gz" | "arc.gz" | ".warc":
                reader_type = "warc"
            case other:
                console.log(f'[red]Could not find a matching reader for file extension "{other}"')
                sys.exit(-1)
    return reader_class_from_name(reader_type)(data_folder, **kwargs)


def find_document_by_id(reader_factory_func, target_id):
    """Find a document with the given ID by iterating through the dataset"""
    reader = reader_factory_func()
    for doc in reader():
        if doc.id == target_id:
            return doc
    return None


def format_metadata(metadata):
    """Format metadata dictionary for display"""
    if not metadata:
        return "[dim]No metadata[/dim]"
    
    lines = []
    for key, value in metadata.items():
        # Truncate long values
        str_value = str(value)
        if len(str_value) > 100:
            str_value = str_value[:97] + "..."
        lines.append(f"[blue]{key}:[/blue] {str_value}")
    return "\n".join(lines)


def get_key():
    """Get a single keypress from stdin"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key


def scrollable_diff_viewer(differences, doc1, doc2):
    """Interactive scrollable diff viewer with j/k navigation"""
    if not differences:
        console.print("[green]No differences found[/green]")
        return
    
    # Prepare all diff lines
    all_diff_lines = []
    
    for diff_type, diff_content in differences:
        if diff_type == "text":
            all_diff_lines.append(("[bold]Text differences:[/bold]", "header"))
            for line in diff_content:
                if line.startswith('---') or line.startswith('+++'):
                    all_diff_lines.append((line.rstrip(), "file_header"))
                elif line.startswith('-'):
                    all_diff_lines.append((line.rstrip(), "removed"))
                elif line.startswith('+'):
                    all_diff_lines.append((line.rstrip(), "added"))
                elif line.startswith('@@'):
                    all_diff_lines.append((line.rstrip(), "hunk_header"))
                else:
                    all_diff_lines.append((line.rstrip(), "context"))
        elif diff_type == "metadata":
            all_diff_lines.append(("[bold]Metadata differences:[/bold]", "header"))
            for diff_line in diff_content:
                all_diff_lines.append((diff_line, "metadata"))
        else:
            all_diff_lines.append((f"[bold]{diff_type}:[/bold] {diff_content}", "other"))
    
    if not all_diff_lines:
        console.print("[green]No differences to display[/green]")
        return
    
    # Display settings
    lines_per_page = console.size.height - 10  # Leave space for headers and controls
    current_line = 0
    total_lines = len(all_diff_lines)
    
    def render_page():
        """Render the current page of diff content"""
        content = Text()
        
        # Add document info header
        content.append(f"Comparing documents: {doc1.id}\n", style="yellow bold")
        content.append(f"Showing lines {current_line + 1}-{min(current_line + lines_per_page, total_lines)} of {total_lines}\n\n", style="dim")
        
        # Add visible diff lines
        end_line = min(current_line + lines_per_page, total_lines)
        for i in range(current_line, end_line):
            line_text, line_type = all_diff_lines[i]
            
            if line_type == "header":
                content.append(line_text + "\n", style="bold")
            elif line_type == "file_header":
                content.append(line_text + "\n", style="bold blue")
            elif line_type == "removed":
                content.append(line_text + "\n", style="red")
            elif line_type == "added":
                content.append(line_text + "\n", style="green")
            elif line_type == "hunk_header":
                content.append(line_text + "\n", style="cyan")
            elif line_type == "metadata":
                content.append(line_text + "\n")
            else:
                content.append(line_text + "\n")
        
        # Add navigation help
        content.append("\n" + "─" * 60 + "\n", style="dim")
        content.append("Navigation: j=down, k=up, q=quit to next document, Ctrl+C=exit\n", style="dim italic")
        
        return content
    
    # Interactive navigation loop
    console.print("Use j/k to scroll through diff, q to continue to next document, Ctrl+C to exit")
    
    try:
        while True:
            # Clear screen and show current page
            console.clear()
            console.print(render_page())
            
            # Get user input
            key = get_key()
            
            if key.lower() == 'j':  # Down
                if current_line + lines_per_page < total_lines:
                    current_line += 1
            elif key.lower() == 'k':  # Up
                if current_line > 0:
                    current_line -= 1
            elif key.lower() == 'q':  # Quit to next document
                break
            elif key == '\x03':  # Ctrl+C
                raise KeyboardInterrupt
            
    except KeyboardInterrupt:
        raise  # Re-raise to handle in main loop


def compare_documents(doc1, doc2, max_diff_lines=None):
    """Compare two documents and return differences"""
    differences = []
    
    # Compare text (don't limit diff lines here since we'll handle scrolling)
    if doc1.text != doc2.text:
        text_diff = list(unified_diff(
            doc1.text.splitlines(keepends=True),
            doc2.text.splitlines(keepends=True),
            fromfile="Dataset 1",
            tofile="Dataset 2",
            n=3
        ))
        if text_diff:
            differences.append(("text", text_diff))
    
    # Compare metadata
    if doc1.metadata != doc2.metadata:
        # Find differing metadata keys
        all_keys = set(doc1.metadata.keys()) | set(doc2.metadata.keys())
        metadata_diffs = []
        
        for key in sorted(all_keys):
            val1 = doc1.metadata.get(key, "[missing]")
            val2 = doc2.metadata.get(key, "[missing]")
            if val1 != val2:
                metadata_diffs.append(f"[blue]{key}:[/blue] [red]{val1}[/red] → [green]{val2}[/green]")
        
        if metadata_diffs:
            differences.append(("metadata", metadata_diffs))
    
    # Compare media (basic comparison)
    if len(doc1.media) != len(doc2.media):
        differences.append(("media_count", f"Media count differs: {len(doc1.media)} vs {len(doc2.media)}"))
    
    return differences


def display_comparison(doc1, doc2, differences, max_diff_lines):
    """Display side-by-side comparison of two documents"""
    
    # Create panels for each document
    doc1_content = Text()
    doc2_content = Text()
    
    # Add basic info
    doc1_content.append(f"ID: {doc1.id}\n", style="yellow")
    doc1_content.append(f"Text length: {len(doc1.text)} chars\n", style="dim")
    doc1_content.append(f"Media count: {len(doc1.media)}\n", style="dim")
    
    doc2_content.append(f"ID: {doc2.id}\n", style="yellow")
    doc2_content.append(f"Text length: {len(doc2.text)} chars\n", style="dim")
    doc2_content.append(f"Media count: {len(doc2.media)}\n", style="dim")
    
    # Show metadata if it differs
    if any(diff_type == "metadata" for diff_type, _ in differences):
        doc1_content.append("\nMetadata:\n", style="bold")
        doc1_content.append(format_metadata(doc1.metadata))
        doc1_content.append("\n")
        
        doc2_content.append("\nMetadata:\n", style="bold")
        doc2_content.append(format_metadata(doc2.metadata))
        doc2_content.append("\n")
    
    # Create side-by-side panels
    panel1 = Panel(doc1_content, title="[bold blue]Dataset 1", border_style="blue")
    panel2 = Panel(doc2_content, title="[bold green]Dataset 2", border_style="green")
    
    console.print(Columns([panel1, panel2], equal=True))
    console.print()
    
    # Use scrollable diff viewer instead of showing all at once
    scrollable_diff_viewer(differences, doc1, doc2)


def main():
    """Main function"""
    args, extra_args = parser.parse_known_args()
    kwargs = dict(extra_arg.split("=") for extra_arg in extra_args)
    
    # Get data folders
    data_folder1 = get_datafolder(args.path1)
    data_folder2 = get_datafolder(args.path2)
    
    console.print(f'Comparing documents between:\n'
                 f'  Dataset 1: "{data_folder1.path}"\n'
                 f'  Dataset 2: "{data_folder2.path}"\n')
    
    # Create reader factory functions (not instances)
    def create_reader1():
        reader = reader_factory(data_folder1, args.reader1, **kwargs)
        return reader
    
    def create_reader2():
        reader = reader_factory(data_folder2, args.reader2, **kwargs)
        return reader
    
    console.print("Starting comparison...")
    sampler = SamplerFilter(args.sample)
    
    # Statistics
    total_compared = 0
    total_different = 0
    total_missing = 0
    
    console.print(f"\nIterating through dataset 1 (sampling rate: {args.sample})...")
    console.print("For each document, searching for matching ID in dataset 2...")
    console.print("Press Ctrl+C to stop\n")
    
    try:
        # Iterate through first dataset with sampling
        reader1 = create_reader1()
        sampled_reader1 = sampler(reader1())
        
        for doc1 in sampled_reader1:
            total_compared += 1
            console.print(f"[dim]Searching for ID: {doc1.id}[/dim]")
            
            # Find matching document in dataset 2 by iterating from start
            doc2 = find_document_by_id(create_reader2, doc1.id)
            
            if doc2 is None:
                total_missing += 1
                console.print(f"[red]Missing in dataset 2:[/red] ID {doc1.id}")
                continue
            
            console.print(f"[green]Found matching document:[/green] ID {doc1.id}")
            
            # Compare documents
            differences = compare_documents(doc1, doc2, args.max_diff_lines)
            
            if differences:
                total_different += 1
                display_comparison(doc1, doc2, differences, args.max_diff_lines)
                
                # Ask user if they want to continue
                if not Confirm.ask("Continue to next document?", default=True):
                    break
            else:
                if not args.skip_identical:
                    console.print(f"[green]✓[/green] ID {doc1.id}: Documents are identical")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Comparison interrupted by user[/yellow]")
    
    # Print summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"Total documents compared: {total_compared}")
    console.print(f"Documents with differences: {total_different}")
    console.print(f"Documents missing in dataset 2: {total_missing}")
    console.print(f"Identical documents: {total_compared - total_different - total_missing}")


if __name__ == "__main__":
    main()