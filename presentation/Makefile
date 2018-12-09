all: presentation.pdf

presentation.pdf: presentation.md header.tex
	pandoc presentation.md -o presentation.pdf -t beamer -H header.tex
