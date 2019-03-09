all: mp

mp:
	pdflatex main.tex
	bibtex main
	bibtex main
	pdflatex main.tex
	pdflatex main.tex
	pdflatex main.tex

clean:
	rm -f main.pdf main.aux main.aux main.bbl main.blg main.log main.out
