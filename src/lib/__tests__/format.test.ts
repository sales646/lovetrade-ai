import { describe, it, expect } from "vitest";
import {
  safeNumber,
  percentOrDash,
  currencyOrDash,
  compactNumber,
  isValidNumber,
} from "../format";

describe("safeNumber", () => {
  it("formats valid numbers correctly", () => {
    expect(safeNumber(123.456)).toBe("123.46");
    expect(safeNumber(123.456, { decimals: 1 })).toBe("123.5");
    expect(safeNumber(100, { prefix: "$" })).toBe("$100.00");
    expect(safeNumber(50, { suffix: "%" })).toBe("50.00%");
  });

  it("returns fallback for invalid values", () => {
    expect(safeNumber(null)).toBe("—");
    expect(safeNumber(undefined)).toBe("—");
    expect(safeNumber(NaN)).toBe("—");
    expect(safeNumber(Infinity)).toBe("—");
    expect(safeNumber(-Infinity)).toBe("—");
  });

  it("uses custom fallback when provided", () => {
    expect(safeNumber(null, { fallback: "N/A" })).toBe("N/A");
  });
});

describe("percentOrDash", () => {
  it("formats percentages with sign", () => {
    expect(percentOrDash(5.25)).toBe("+5.25%");
    expect(percentOrDash(-3.75)).toBe("-3.75%");
    expect(percentOrDash(0)).toBe("+0.00%");
  });

  it("returns dash for invalid values", () => {
    expect(percentOrDash(null)).toBe("—");
    expect(percentOrDash(undefined)).toBe("—");
    expect(percentOrDash(NaN)).toBe("—");
  });

  it("respects decimal precision", () => {
    expect(percentOrDash(5.12345, 1)).toBe("+5.1%");
    expect(percentOrDash(5.12345, 3)).toBe("+5.123%");
  });
});

describe("currencyOrDash", () => {
  it("formats currency correctly", () => {
    expect(currencyOrDash(1234.56)).toBe("$1234.56");
    expect(currencyOrDash(1234.56, "€")).toBe("€1234.56");
  });

  it("returns dash for invalid values", () => {
    expect(currencyOrDash(null)).toBe("—");
    expect(currencyOrDash(undefined)).toBe("—");
  });
});

describe("compactNumber", () => {
  it("formats large numbers with suffixes", () => {
    expect(compactNumber(1500)).toBe("1.5K");
    expect(compactNumber(1500000)).toBe("1.5M");
    expect(compactNumber(1500000000)).toBe("1.5B");
  });

  it("handles negative numbers", () => {
    expect(compactNumber(-1500)).toBe("-1.5K");
    expect(compactNumber(-1500000)).toBe("-1.5M");
  });

  it("returns dash for invalid values", () => {
    expect(compactNumber(null)).toBe("—");
    expect(compactNumber(undefined)).toBe("—");
    expect(compactNumber(NaN)).toBe("—");
  });

  it("respects decimal precision", () => {
    expect(compactNumber(1234, 2)).toBe("1.23K");
    expect(compactNumber(1234, 0)).toBe("1K");
  });
});

describe("isValidNumber", () => {
  it("returns true for valid numbers", () => {
    expect(isValidNumber(0)).toBe(true);
    expect(isValidNumber(123.456)).toBe(true);
    expect(isValidNumber(-789)).toBe(true);
  });

  it("returns false for invalid values", () => {
    expect(isValidNumber(null)).toBe(false);
    expect(isValidNumber(undefined)).toBe(false);
    expect(isValidNumber(NaN)).toBe(false);
    expect(isValidNumber(Infinity)).toBe(false);
    expect(isValidNumber("123")).toBe(false);
    expect(isValidNumber({})).toBe(false);
  });
});
