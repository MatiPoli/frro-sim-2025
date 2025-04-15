import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Simulación de ruleta")
    parser.add_argument("--tiradas", type=int, default=100, help="Cantidad de tiradas por corrida")
    parser.add_argument("--corridas", type=int, default=10, help="Cantidad de corridas a realizar")
    return parser.parse_args()