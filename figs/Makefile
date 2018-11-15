all: model-overview.png clean

model-overview.png: model-overview.tex
	pdflatex model-overview.tex
	pdftoppm model-overview.pdf model-overview.png -png
	mv model-overview.png-1.png model-overview.png
	rm model-overview.pdf

clean:
	rm -rf *.log *.aux *.bib
