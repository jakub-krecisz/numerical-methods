DIR = $(notdir $(CURDIR))

show_plots:
	python3 main.py show

save_plots:
	python3 main.py save

clean:
	rm -rf __pycache__

tar: clean
	(cd ../; tar -cvzf $(DIR).tar.gz $(DIR))

