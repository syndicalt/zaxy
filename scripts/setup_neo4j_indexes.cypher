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
CREATE CONSTRAINT entity_version_id IF NOT EXISTS
FOR (e:Entity)
REQUIRE (e.session_id, e.name, e.entity_type, e.valid_from) IS UNIQUE;

CREATE CONSTRAINT session_id IF NOT EXISTS
FOR (s:Session)
REQUIRE s.id IS UNIQUE;

CREATE CONSTRAINT event_identity IF NOT EXISTS
FOR (ev:Event)
REQUIRE (ev.session_id, ev.seq) IS UNIQUE;

// ------------------------------------------------------------------
// Provenance lookup indexes
// ------------------------------------------------------------------
CREATE INDEX entity_lookup IF NOT EXISTS
FOR (e:Entity)
ON (e.session_id, e.name, e.entity_type);

CREATE INDEX event_hash IF NOT EXISTS
FOR (ev:Event)
ON (ev.session_id, ev.hash);

CREATE INDEX event_prev_hash IF NOT EXISTS
FOR (ev:Event)
ON (ev.session_id, ev.prev_hash);

// ------------------------------------------------------------------
// Verify
// ------------------------------------------------------------------
SHOW INDEXES;
SHOW CONSTRAINTS;
