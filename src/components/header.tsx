import { Button } from "@/components/ui/button"
import { ThemeToggle } from "./theme-toggle"
import { Bot, Menu, X } from "lucide-react"
import { useState } from "react"

export function Header() {
  const [isMenuOpen, setIsMenuOpen] = useState(false)

  const navItems = [
    { label: "Services", href: "#services" },
    { label: "About", href: "#about" },
    { label: "Contact", href: "#contact" }
  ]

  const scrollToSection = (href: string) => {
    const element = document.querySelector(href)
    element?.scrollIntoView({ behavior: 'smooth' })
    setIsMenuOpen(false)
  }

  return (
    <header className="fixed top-0 w-full z-50 border-b border-border/50 bg-background/80 backdrop-blur-md">
      <div className="container mx-auto px-6">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-gradient-primary rounded-lg flex items-center justify-center">
              <Bot className="h-4 w-4 text-white" />
            </div>
            <span className="font-mono font-bold text-lg">AI.Dev</span>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center gap-8">
            {navItems.map((item) => (
              <button
                key={item.label}
                onClick={() => scrollToSection(item.href)}
                className="font-mono text-sm hover:text-primary transition-colors"
              >
                {item.label}
              </button>
            ))}
          </nav>

          {/* Actions */}
          <div className="flex items-center gap-4">
            <ThemeToggle />
            <Button 
              size="sm" 
              className="hidden md:flex bg-gradient-primary hover:shadow-glow transition-all duration-300 font-mono"
              onClick={() => scrollToSection('#contact')}
            >
              Hire Me
            </Button>
            
            {/* Mobile Menu Button */}
            <Button
              variant="ghost"
              size="sm"
              className="md:hidden"
              onClick={() => setIsMenuOpen(!isMenuOpen)}
            >
              {isMenuOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
            </Button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {isMenuOpen && (
          <div className="md:hidden border-t border-border/50 py-4">
            <nav className="flex flex-col gap-4">
              {navItems.map((item) => (
                <button
                  key={item.label}
                  onClick={() => scrollToSection(item.href)}
                  className="font-mono text-sm hover:text-primary transition-colors text-left"
                >
                  {item.label}
                </button>
              ))}
              <Button 
                size="sm" 
                className="mt-2 bg-gradient-primary hover:shadow-glow transition-all duration-300 font-mono w-fit"
                onClick={() => scrollToSection('#contact')}
              >
                Hire Me
              </Button>
            </nav>
          </div>
        )}
      </div>
    </header>
  )
}