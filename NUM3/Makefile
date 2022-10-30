DIR = $(notdir $(CURDIR))

run:
	python3 main.py

clean:
	rm -rf __pycache__

tar: clean
	(cd ../; tar -cvzf $(DIR).tar.gz $(DIR))
