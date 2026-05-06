#!/usr/bin/env bash
# Generate self-signed certificates for Neo4j TLS.
# Usage: ./scripts/generate-certs.sh [output_dir]

set -euo pipefail

OUTPUT_DIR="${1:-.certs}"
mkdir -p "${OUTPUT_DIR}"

echo "🔐 Generating self-signed certificates for Neo4j..."

# Generate CA key and cert
openssl req -x509 -newkey rsa:4096 -sha256 -days 365 \
    -nodes -keyout "${OUTPUT_DIR}/ca.key" -out "${OUTPUT_DIR}/ca.crt" \
    -subj "/C=US/ST=State/L=City/O=Zaxy/CN=Zaxy Neo4j CA" \
    2>/dev/null

# Generate server key and CSR
openssl req -newkey rsa:4096 -nodes \
    -keyout "${OUTPUT_DIR}/server.key" -out "${OUTPUT_DIR}/server.csr" \
    -subj "/C=US/ST=State/L=City/O=Zaxy/CN=neo4j" \
    2>/dev/null

# Sign server cert with CA
cat > "${OUTPUT_DIR}/server.ext" <<EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = neo4j
IP.1 = 127.0.0.1
EOF

openssl x509 -req -in "${OUTPUT_DIR}/server.csr" -CA "${OUTPUT_DIR}/ca.crt" \
    -CAkey "${OUTPUT_DIR}/ca.key" -CAcreateserial -out "${OUTPUT_DIR}/server.crt" \
    -days 365 -sha256 -extfile "${OUTPUT_DIR}/server.ext" \
    2>/dev/null

# Combine for Neo4j Bolt SSL policy.
# Neo4j expects private.key, public.crt, trusted/, and revoked/ under the policy base directory.
mkdir -p "${OUTPUT_DIR}/neo4j/trusted" "${OUTPUT_DIR}/neo4j/revoked"
cp "${OUTPUT_DIR}/server.key" "${OUTPUT_DIR}/neo4j/private.key"
cp "${OUTPUT_DIR}/server.crt" "${OUTPUT_DIR}/neo4j/public.crt"
cp "${OUTPUT_DIR}/ca.crt" "${OUTPUT_DIR}/neo4j/trusted/public.crt"
chmod 644 "${OUTPUT_DIR}/neo4j/private.key" "${OUTPUT_DIR}/neo4j/public.crt" "${OUTPUT_DIR}/neo4j/trusted/public.crt"

# Cleanup intermediates
rm -f "${OUTPUT_DIR}/server.csr" "${OUTPUT_DIR}/server.ext" "${OUTPUT_DIR}/ca.srl"

echo "✅ Certificates generated in ${OUTPUT_DIR}/"
echo "   CA cert: ${OUTPUT_DIR}/ca.crt"
echo "   Server cert: ${OUTPUT_DIR}/server.crt"
echo "   Server key: ${OUTPUT_DIR}/server.key"
echo "   Neo4j bundle: ${OUTPUT_DIR}/neo4j/"
