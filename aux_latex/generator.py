import os
import json
import pandas as pd
# Diretório dos resultados dos testes
results_directory = "test/results"

# Itera sobre cada arquivo no diretório
for foldername in os.listdir(results_directory):
    # Obtém o caminho completo do diretório do teste
    folder_path = os.path.join(results_directory, foldername)
    
    json_file_path = f'{folder_path}/info.json'
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        print(data)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except FileNotFoundError:
        print(f"File not found: {json_file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
    # print(data)

    # Verifica se é um diretório
    if os.path.isdir(folder_path):
        
        if os.path.isdir(folder_path) and foldername.startswith("test"):
            report_df = pd.read_csv(f"{folder_path}/report.csv")
            report_latex = report_df.to_latex(index=False)
        
        # Gera o texto para o teste
            text = f"""% ----------------------------------------------------------
% Teste {foldername}
% ----------------------------------------------------------
\\subsubsection{{Teste {foldername} - AlexNet (Is That a Santa)}}

Informações utilizadas para o treinamento.

\\begin{{table}}[ht]
   \\centering
   \\caption{{Treinamento}}
   \\label{{tab:modelos}}
   \\begin{{tabular}}{{| c | c | }}
      \\hline 
      \\textbf{{Informação}} & \\textbf{{Descrição}} \\\\
      \\hline \\hline 
      Rede & AlexNet \\\\
      \\hline
      Número de épocas & {data['epoca']}\\\\
      \\hline
      Tamanho do lote & {data['lote']}\\\\
      \\hline
      Taxa inicial & {data['tx_inical']} \\\\
      \\hline
      Taxa de decaimento & {data['decaimento']} \\\\
      \\hline
      Total de classes & {data['total_classes']}\\\\
      \\hline
      Dataset & CIFAR-10\\\\
      \\hline
   \\end{{tabular}} 
\\end{{table}}

Resultados obtidos após treinamento.

{report_latex}

\\begin{{figure}}[ht]
 \\begin{{center}}
   \\includegraphics[scale=1]{{tests/{foldername}/confusion_matrix.png}}
  \\caption{{Matriz de Confusão}}
  \\label{{fig:fig03}}
 \\end{{center}}
\\end{{figure}}

\\begin{{figure}}[ht]
 \\begin{{center}}
   \\includegraphics[scale=0.8]{{tests/{foldername}/loss_over_time.png}}
  \\caption{{Gráfico de Perda}}
  \\label{{fig:fig04}}
 \\end{{center}}
\\end{{figure}}
"""

            # Salva o texto em um arquivo no mesmo diretório
            output_file_path = os.path.join(folder_path, f"{foldername}_result.tex")
            with open(output_file_path, "w") as output_file:
                output_file.write(text)

print("Script concluído.")
