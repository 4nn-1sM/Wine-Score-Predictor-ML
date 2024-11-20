

# Función para convertir puntos en categorías
def categorize_points(points): 
    if 98 <= points <= 100:
        result = "Clásico - El pináculo de la calidad (98-100)"
    elif 94 <= points < 98:
        result = "Excepcional - Un gran logro (94-97)"
    elif 90 <= points < 94:
        result = "Excelente - Altamente recomendado (90-93)"
    elif 87 <= points < 90:
        result = "Muy bueno - A menudo buena relación calidad-precio; bien recomendado (87-89)"
    elif 83 <= points < 87:
        result = "Bueno - Adecuado para el consumo diario, a menudo buena relación calidad-precio (83-86)"
    elif 80 <= points < 83:
        result = "Aceptable - Puede ser empleado (80-82)"
    else:
        result = "Sin calificar"
    return result