"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const handler = require("../api/advisory");

function invoke(query = {}, method = "GET") {
  return new Promise((resolve, reject) => {
    const headers = {};
    const res = {
      statusCode: 0,
      setHeader(key, value) {
        headers[key] = value;
      },
      end(body) {
        try {
          resolve({
            status: this.statusCode,
            headers,
            json: body ? JSON.parse(body) : null,
          });
        } catch (error) {
          reject(error);
        }
      },
    };
    Promise.resolve(handler({ method, query }, res)).catch(reject);
  });
}

test("female adult shows lactation, milk and mastitis advice", async () => {
  const response = await invoke({
    species: "Cattle",
    breed: "gir",
    gender: "Female",
    age: "Adult",
    lang: "en",
  });
  assert.equal(response.status, 200);
  assert.equal(response.json.ok, true);
  assert.equal(response.json.breed.slug, "gir");
  assert.equal(response.json.advisory.context.milk_yield_visible, true);
  assert.equal(response.json.advisory.context.draught_power_visible, false);
  const ids = response.json.advisory.cards.map((card) => card.id);
  assert.ok(ids.includes("lactation_diet"));
  assert.ok(ids.includes("milk_yield"));
  assert.ok(ids.includes("mastitis"));
  assert.match(response.json.advisory.cards.find((card) => card.id === "milk_yield").body, /8 - 12 L\/day/);
});

test("male adult hides milk and shows draught advice", async () => {
  const response = await invoke({
    species: "Cattle",
    breed: "hallikar",
    gender: "Male",
    age: "Adult",
    lang: "en",
  });
  assert.equal(response.status, 200);
  assert.equal(response.json.advisory.context.milk_yield_visible, false);
  assert.equal(response.json.advisory.context.draught_power_visible, true);
  const ids = response.json.advisory.cards.map((card) => card.id);
  assert.ok(ids.includes("draught"));
  assert.ok(ids.includes("breeding_value"));
  assert.ok(!ids.includes("milk_yield"));
});

test("calf hides milk and draught and returns a vaccination schedule", async () => {
  const response = await invoke({
    species: "Buffalo",
    breed: "murrah",
    gender: "Male",
    age: "Calf",
    lang: "en",
  });
  assert.equal(response.status, 200);
  assert.equal(response.json.advisory.context.milk_yield_visible, false);
  assert.equal(response.json.advisory.context.draught_power_visible, false);
  assert.ok(response.json.advisory.vaccination_schedule.items.length >= 4);
  const ids = response.json.advisory.cards.map((card) => card.id);
  assert.ok(ids.includes("colostrum"));
  assert.ok(ids.includes("vaccination"));
});

test("Tamil and Hindi responses contain translated copy", async () => {
  const ta = await invoke({
    species: "Cattle",
    breed: "gir",
    gender: "Female",
    age: "Adult",
    lang: "ta",
  });
  const hi = await invoke({
    species: "Cattle",
    breed: "gir",
    gender: "Female",
    age: "Adult",
    lang: "hi",
  });
  assert.equal(ta.json.breed.name, "கிர்");
  assert.match(ta.json.advisory.title, /பால்/);
  assert.match(hi.json.advisory.title, /दुग्ध/);
  assert.equal(hi.json.breed.name, "गीर");
});

test("aliases resolve and unknown breeds return 404", async () => {
  const alias = await invoke({
    species: "Buffalo",
    breed: "bhadawari",
    gender: "Female",
    age: "Adult",
    lang: "en",
  });
  assert.equal(alias.status, 200);
  assert.equal(alias.json.breed.slug, "bhadwari");

  const missing = await invoke({
    species: "Cattle",
    breed: "not_a_breed",
    gender: "Female",
    age: "Adult",
    lang: "en",
  });
  assert.equal(missing.status, 404);
  assert.equal(missing.json.error, "unknown_breed");
});

test("CORS and method handling", async () => {
  const options = await invoke({}, "OPTIONS");
  assert.equal(options.status, 204);
  assert.equal(options.headers["Access-Control-Allow-Origin"], "*");

  const post = await invoke({}, "POST");
  assert.equal(post.status, 405);

  const missing = await invoke({ species: "Cattle" });
  assert.equal(missing.status, 400);
});

test("knowledge base contains exactly 50 cattle and 18 buffalo", async () => {
  const response = await invoke({
    species: "Cattle",
    breed: "gir",
    gender: "Female",
    age: "Adult",
    lang: "en",
  });
  assert.equal(response.json.meta.breed_count, 68);
  assert.equal(response.json.meta.counts.Cattle, 50);
  assert.equal(response.json.meta.counts.Buffalo, 18);
});
