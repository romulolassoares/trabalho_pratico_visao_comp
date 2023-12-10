import subprocess

# Lista de scripts a serem executados
scripts = ["test01vgg.py"]

# Itera sobre a lista de scripts e os executa
for script in scripts:
    try:
        # Executa o script usando subprocess
        subprocess.run(["python", script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar {script}: {e}")