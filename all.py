import os

# --- Konfigurasjon ---
OUTPUT_FILENAME = "prosjekt_struktur_og_innhold.txt"
ALLOWED_EXTENSIONS = ['.py', '.json', '.txt', '.md']  # Filtyper som skal inkluderes
MAX_FILE_SIZE_BYTES = 100 * 1024  # Maks 100 KB for å unngå store filer
EXCLUDE_DIRS = [
    '.git',
    '__pycache__',
    '.vscode',
    '.idea',
    'venv',  # Vanlig navn for Python virtual environments
    'env',  # Annet vanlig navn
    '.venv',  # Annet vanlig navn (ofte skjult)
    'node_modules'  # Viktig for frontend-prosjekter
]  # Mapper som skal ignoreres fullstendig
# Få navnet på denne skriptfilen for å ekskludere den
THIS_SCRIPT_NAME = os.path.basename(__file__)


# --- Slutt Konfigurasjon ---

def get_project_structure(start_path_abs):
    """
    Genererer en streng som representerer mappestrukturen og filinnhold.
    `start_path_abs` må være en absolutt sti.
    """
    output_lines = []
    output_lines.append(f"Prosjektstruktur og filinnhold (fra mappen: {start_path_abs})\n")
    output_lines.append(f"Skriptet '{THIS_SCRIPT_NAME}' og outputfilen '{OUTPUT_FILENAME}' er ekskludert.")
    output_lines.append(
        f"Følgende mappenavn blir også fullstendig ignorert (i tillegg til andre skjulte mapper som starter med '.'): {', '.join(EXCLUDE_DIRS)}\n")
    output_lines.append("=" * 80 + "\n")

    # Absolutte stier for skriptet og outputfilen for nøyaktig ekskludering
    script_file_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), THIS_SCRIPT_NAME))
    output_file_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), OUTPUT_FILENAME))

    for root, dirs, files_in_dir in os.walk(start_path_abs, topdown=True):
        # Fjern mapper som skal ekskluderes fra videre traversering
        # Også fjern alle mapper som starter med '.' (skjulte mapper generelt)
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.')]

        # Filtrer filer før sortering og prosessering
        actual_files_to_process = []
        for f_loopvar in files_in_dir:
            current_file_abs_path = os.path.abspath(os.path.join(root, f_loopvar))

            if current_file_abs_path == script_file_abs_path:
                continue
            if current_file_abs_path == output_file_abs_path:
                continue

            if f_loopvar.startswith('.'):  # Hopp over andre skjulte filer
                continue
            actual_files_to_process.append(f_loopvar)

        actual_files_to_process.sort()  # Sorter den filtrerte listen

        # Beregn relativ sti fra startpunktet for penere output og nivåbestemmelse
        # Hvis root er den samme som start_path_abs, vil rel_root være '.'
        rel_root = os.path.relpath(root, start_path_abs)

        level = 0
        if rel_root != ".":
            level = rel_root.count(os.sep)
            indent_for_dir = ' ' * 4 * level
            output_lines.append(f"{indent_for_dir}{os.path.basename(root)}/")

        # Filer får ett nivå mer innrykk enn sin mappe (eller fra roten hvis i rotmappen)
        indent_for_file = ' ' * 4 * (level + 1)

        for f_name in actual_files_to_process:
            file_path_for_reading = os.path.join(root, f_name)
            _, ext = os.path.splitext(f_name)
            ext = ext.lower()

            if ext in ALLOWED_EXTENSIONS:
                output_lines.append(f"{indent_for_file}|-- {f_name}")
                try:
                    file_size = os.path.getsize(file_path_for_reading)
                    if file_size == 0:
                        output_lines.append(f"{indent_for_file}    [--- INNHOLD: Tom fil ---]")
                    elif file_size > MAX_FILE_SIZE_BYTES:
                        output_lines.append(
                            f"{indent_for_file}    [--- INNHOLD UTELATT: Filstørrelse ({file_size / 1024:.2f} KB) > maks ({MAX_FILE_SIZE_BYTES / 1024:.0f} KB) ---]")
                    else:
                        with open(file_path_for_reading, 'r', encoding='utf-8', errors='ignore') as f_content:
                            content = f_content.read().strip()  # .strip() fjerner ledende/etterfølgende tomrom

                        if not content:  # Sjekk om filen ble tom etter strip()
                            output_lines.append(
                                f"{indent_for_file}    [--- INNHOLD: Filen inneholder kun tomrom eller er tom ---]")
                        else:
                            output_lines.append(f"{indent_for_file}    --- START INNHOLD ({f_name}) ---")
                            for line_content in content.splitlines():
                                output_lines.append(f"{indent_for_file}    {line_content}")
                            output_lines.append(f"{indent_for_file}    --- SLUTT INNHOLD ({f_name}) ---")
                except Exception as e:
                    output_lines.append(f"{indent_for_file}    [--- FEIL VED LESING AV FIL: {e} ---]")
                output_lines.append("")  # Tom linje for lesbarhet mellom filer

    return "\n".join(output_lines)


if __name__ == "__main__":
    # Mappen der dette skriptet ligger, vil være rot for skanningen
    current_script_dir_abs = os.path.dirname(os.path.abspath(__file__))
    output_file_path_abs = os.path.join(current_script_dir_abs, OUTPUT_FILENAME)

    if THIS_SCRIPT_NAME == OUTPUT_FILENAME:
        print(
            f"FEIL: Skriptnavnet ({THIS_SCRIPT_NAME}) kan ikke være det samme som outputfilnavnet ({OUTPUT_FILENAME}).")
        print("Vennligst endre OUTPUT_FILENAME i skriptet.")
    else:
        try:
            project_info = get_project_structure(current_script_dir_abs)
            with open(output_file_path_abs, 'w', encoding='utf-8') as f_out:
                f_out.write(project_info)
            print(f"Mappestruktur og filinnhold lagret i: {output_file_path_abs}")
        except Exception as e:
            print(f"En feil oppstod under generering av filen: {e}")