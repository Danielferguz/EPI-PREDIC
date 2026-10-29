#!/usr/bin/env python3
"""Genera SAL, ITER y HUELLA para auth.js. Uso: python3 generar_huella.py usuario clave"""
import os, sys, hashlib
u, p = sys.argv[1].strip().lower(), sys.argv[2]; sal = os.urandom(16); it = 200000
print("var SAL = '%s', ITER = %d, HUELLA = '%s';" % (sal.hex(), it, hashlib.pbkdf2_hmac('sha256', (u + '\0' + p).encode(), sal, it, 32).hex()))
