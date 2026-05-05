// Zaxy Neo4j Index Setup
// Run this in Neo4j Browser or via cypher-shell after starting Neo4j.

// ------------------------------------------------------------------
// Full-text index for keyword/BM25 search
// ------------------------------------------------------------------
CREATE FULLTEXT INDEX entity_fulltext FOR (e:Entity)
ON EACH [e.name, e.summary];

// ------------------------------------------------------------------
// Vector index for embedding similarity search
// Requires: Neo4j 5.11+ with GDS plugin or native vector support
// ------------------------------------------------------------------
CREATE VECTOR INDEX entity_vector FOR (e:Entity)
ON e.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine'
}};

// ------------------------------------------------------------------
// Constraints
// ------------------------------------------------------------------
CREATE CONSTRAINT entity_unique IF NOT EXISTS
FOR (e:Entity)
REQUIRE (e.name, e.entity_type) IS NODE KEY;

// ------------------------------------------------------------------
// Verify
// ------------------------------------------------------------------
SHOW INDEXES;
SHOW CONSTRAINTS;
