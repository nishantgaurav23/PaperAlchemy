import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import Home from "./page";

vi.mock("next-themes", () => ({
  useTheme: () => ({ theme: "light", setTheme: vi.fn() }),
}));

describe("Home Page", () => {
  it("renders the heading", () => {
    render(<Home />);
    expect(screen.getByText("PaperAlchemy")).toBeInTheDocument();
  });

  it("renders the description", () => {
    render(<Home />);
    expect(
      screen.getByText(/AI Research Assistant/),
    ).toBeInTheDocument();
  });
});
