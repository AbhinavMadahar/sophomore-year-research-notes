all: README.pdf

README.pdf: README.md header.tex
	pandoc README.md -o README.pdf -H header.tex
