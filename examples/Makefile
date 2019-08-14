

default:
	@echo "make webp?"

prepare:
	python3 lambda.py

apng: prepare
	# /kc is essential for the frames not to overlap
	apngasm out.png step*.png /kc /kp

gif: prepare
	# <https://stackoverflow.com/a/23285387/353337>
	convert -dispose 2 -delay 10 -loop 0 smoothfit-lambda*.png out.gif

webp: prepare
	img2webp smoothfit-lambda*.png -min_size -lossy -o out.webp

clean:
	rm -f *.png *.webp *.gif
