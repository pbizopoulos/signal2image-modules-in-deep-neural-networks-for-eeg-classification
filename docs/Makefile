.POSIX:

all: tmp tmp/all-done

check: tmp tmp/check-done

clean:
	rm -rf result tmp/

.gitignore:
	printf '.env\nresult\ntmp/\n' > $@

flake.nix:
	nix flake init

index.html:
	touch $@

tmp:
	mkdir $@

tmp/all-done: .gitignore flake.nix index.html
	nix develop .#all
	touch $@

tmp/check-done: .gitignore flake.nix index.html
	nix develop .#check
	touch $@
