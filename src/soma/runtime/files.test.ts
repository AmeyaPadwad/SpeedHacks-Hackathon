import { describe, expect, it } from "vitest";
import { FileRefIndex, splitRef } from "./files";

describe("FileRefIndex", () => {
  const idx = new FileRefIndex(["f1/app/calc.py", "f1/src/main.py", "f2/main.py", "bad", "f1/../x.py"]);

  it("resolves absolute, relative and bare paths by trailing components", () => {
    expect(idx.resolve("C:\\proj\\app\\calc.py")).toBe("f1/app/calc.py");
    expect(idx.resolve("app/calc.py")).toBe("f1/app/calc.py");
    expect(idx.resolve("./app/calc.py")).toBe("f1/app/calc.py");
    expect(idx.resolve("calc.py")).toBe("f1/app/calc.py");
    expect(idx.resolve("src/main.py")).toBe("f1/src/main.py");
  });

  it("never guesses between ambiguous files and ignores unknown or traversing paths", () => {
    expect(idx.resolve("main.py")).toBeNull();
    expect(idx.resolve("other.py")).toBeNull();
    expect(idx.resolve("../app/calc.py")).toBeNull();
    expect(idx.resolve("")).toBeNull();
    expect(idx.size).toBe(3);
  });

  it("splitRef", () => {
    expect(splitRef("f1/a/b.py")).toEqual(["f1", "a/b.py"]);
    expect(splitRef("f1/")).toBeNull();
    expect(splitRef("/a")).toBeNull();
    expect(splitRef("f1/a/../b")).toBeNull();
  });
});
