all: mp

mp:
	pdflatex main.tex
	bibtex main
	bibtex main
	pdflatex main.tex
	pdflatex main.tex
	pdflatex main.tex

tr:
	pdflatex mainTR.tex
	bibtex mainTR
	bibtex mainTR
	pdflatex mainTR.tex
	pdflatex mainTR.tex
	pdflatex mainTR.tex

clean:
	rm -f main.pdf main.aux main.bbl mainTR.pdf mainTR.aux mainTR.bbl
