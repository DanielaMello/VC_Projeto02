def filter_error_lines(input_file, output_file):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    filtered_lines = []
    for line in lines:
        if line.startswith('#') or line.strip() == '':
            filtered_lines.append(line)
        else:
            parts = line.strip().split()
            if parts[1] != 'err':
                filtered_lines.append(line)

    with open(output_file, 'w') as outfile:
        outfile.writelines(filtered_lines)


# Definindo caminho dos arquivos de entrada e saída
input_file_path = 'C:/Users/danie/Downloads/Projeto/words_original.txt'
output_file_path = 'C:/Users/danie/Downloads/Projeto/words_filtered.txt'

# Filtrar linhas e salvar em um novo arquivo
filter_error_lines(input_file_path, output_file_path)
