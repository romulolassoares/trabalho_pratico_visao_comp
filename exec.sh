#!/bin/bash

# Lista de arquivos Python que começam com "test"
arquivos_python=($(ls test*.py))

# Verifica se há arquivos a serem executados
if [ ${#arquivos_python[@]} -eq 0 ]; then
    echo "Nenhum arquivo Python encontrado."
    exit 1
fi

# Itera sobre cada arquivo Python na lista
for arquivo in "${arquivos_python[@]}"; do
    echo "Executando $arquivo..."
    
    # Executa o arquivo Python
    python "$arquivo"
    
    # Verifica o código de saída
    codigo_saida=$?
    
    # Exibe uma mensagem com base no código de saída
    if [ $codigo_saida -eq 0 ]; then
        echo "Execução de $arquivo concluída com sucesso."
    else
        echo "Erro durante a execução de $arquivo. Código de saída: $codigo_saida"
    fi
    
    echo "----------------------------------------"
done

echo "Todos os arquivos Python foram executados."

