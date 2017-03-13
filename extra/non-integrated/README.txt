Dependencias: libopencv2


Directorios:

+ imgmat: utilidades para transformar imágenes a texto y viceversa y generador de ruido. Escrito en C++.
	- img2mat transforma una imagen a texto(en escala de grises).
		Uso:
			img2mat <imagen> <salida>(matriz)
	- mat2img transforma una matriz a imagen.
		Uso:
			img2mat <matriz> <salida>(imagen)
	- noise_maker genera matrices de ruido 
		Uso:
			noise_maker <matriz> <intervalos> <sigma_min> <sigma_max> <repeticiones>

+ edge_detector: programa para detección de contornos que escribe matrices en disco. Escrito en C.
	Uso:
		edge_detector <matrix_file> <edge_out_file> g <mask_file> (gradient detector) or
		edge_detector <matrix_file> <edge_out_file> c (coefficient of variation detector)
	Salida:
		<matrix_file> <edge_detector> <size> <compute_time> <read_time> <write_time>

+ performance: programa que hace uso de la función de performance para encontrar el mejor contorno.
	Uso:
		perfomance <edge_file> <ground_truth_file> <method> <options>
		method: e(exhaustive) or o(optimized)
	Options:
		-f <filename>: writes output to file specified (by default prints to stdout)
		-a: prints all the calculated values, not only the optimal(only applies to exhaustive method)
	Output:
		<edge_file> <method> <matrix_size> <threshold> <value> <compute_time> <read_time>


Formato de matrices para todos los programas:

<alto(x)> <ancho(y)>
<x1y1> <x1y2> ... <x1yn>
 ...
<xny1> <xny2> ... <xnyn>


Scripts:

+ compile.sh: compila todo(equivalente a hacer "make" en cada directorio)

+ mat2img_dir.sh: convierte todas las matrices de un directorio en imágenes y las almacena en data/img

+ execute.sh: ejecuta todas las pruebas de forma automatizada y muestra tiempos totales por operación
	Uso:
		execute.sh <image> <ground_truth> <output_dir>
		(output_dir se puede omitir, y en tal caso se tomará el directorio actual)
	Dentro del archivo se definen los parámetros de generación de ruido:
	- NOISE_INTS: número de valores de sigma calculadores entre el mínimo y el máximo
	- NOISE_SMIN: sigma mínimo
	- NOISE_SMAX: sigma máximo
	- NOISE_REPS: repeticiones para un mismo sigma
	Igual se define EDGE_MASK para la ruta del archivo que contiene la máscara a usar en el detector de gradiente.
	Las matrices se almacenan en data/matrix/<nombre_imagen> y las salidas de los programas en data/exec/<nombre_imagen>_fecha-hora.