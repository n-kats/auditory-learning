import { clampPage, formatPageLabel, parseJumpPage } from "./pageState";

describe("pageState", () => {
  it("clamps page bounds", () => {
    expect(clampPage(0, 12)).toBe(1);
    expect(clampPage(3.9, 12)).toBe(3);
    expect(clampPage(99, 12)).toBe(12);
  });

  it("parses jump page safely", () => {
    expect(parseJumpPage(" 7 ", 10)).toBe(7);
    expect(parseJumpPage("x", 10)).toBeNull();
    expect(parseJumpPage("", 10)).toBeNull();
  });

  it("formats page label", () => {
    expect(formatPageLabel(3, 18)).toBe("3 / 18");
  });
});
