import numpy as np

class Ruleta:
    def __init__(self):
        self.numeros = np.arange(0, 37)
        self.colores = self._asignar_colores()
        self.columnas = self._asignar_columnas()

    def _asignar_colores(self):
        colores = {0: 'verde'}
        rojos = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}
        for n in self.numeros[1:]:
            colores[n] = 'rojo' if n in rojos else 'negro'
        return colores

    def _asignar_columnas(self):
        columnas = {}
        for n in self.numeros:
            if n == 0:
                columnas[n] = 'ninguna'
            elif n % 3 == 1:
                columnas[n] = '1ra'
            elif n % 3 == 2:
                columnas[n] = '2da'
            else:
                columnas[n] = '3ra'
        return columnas

    def girar(self):
        resultado = np.random.choice(self.numeros)
        color = self.colores[resultado]
        paridad = 'par' if resultado != 0 and resultado % 2 == 0 else 'impar' if resultado != 0 else 'ninguna'
        docena = (
            '1ra' if 1 <= resultado <= 12 else
            '2da' if 13 <= resultado <= 24 else
            '3ra' if 25 <= resultado <= 36 else
            'ninguna'
        )
        mitad = (
            '1ra mitad' if 1 <= resultado <= 18 else
            '2da mitad' if 19 <= resultado <= 36 else
            'ninguna'
        )
        columna = self.columnas[resultado]

        return {
            'numero': resultado,
            'color': color,
            'paridad': paridad,
            'docena': docena,
            'mitad': mitad,
            'columna': columna
        }

# Prueba del simulador
if __name__ == "__main__":
    ruleta = Ruleta()
    resultado = ruleta.girar()
    print(f"🎯 Número: {resultado['numero']}")
    print(f"🎨 Color: {resultado['color']}")
    print(f"➗ Paridad: {resultado['paridad']}")
    print(f"📦 Docena: {resultado['docena']}")
    print(f"🌓 Mitad: {resultado['mitad']}")
    print(f"📊 Columna: {resultado['columna']}")
