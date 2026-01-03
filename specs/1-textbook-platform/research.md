# Research: AI-Native Technical Textbook Platform - Phase 1

## Decision: Docusaurus Framework Choice
**Rationale**: Docusaurus is a mature, well-supported static site generator specifically designed for documentation. It offers built-in features like search, versioning, and responsive design that are ideal for a textbook platform. It supports Markdown content, which aligns with the requirement for markdown-based textbook content.

**Alternatives considered**:
- GitBook: Good for books but less flexible than Docusaurus
- Hugo: More complex setup, primarily for blogs
- Jekyll: Older technology, requires more configuration
- Custom React app: More complex, requires more maintenance

## Decision: GitHub Pages Deployment
**Rationale**: GitHub Pages provides free hosting, integrates seamlessly with Git workflows, and offers custom domain support. It's ideal for static documentation sites and supports the requirement for public accessibility.

**Alternatives considered**:
- Netlify: Requires additional setup, but more features
- Vercel: Good alternative but GitHub Pages is simpler for this use case
- AWS S3: More complex setup, unnecessary for static content

## Decision: Content Structure Organization
**Rationale**: Organizing content by parts and chapters in a hierarchical folder structure follows Docusaurus best practices and makes navigation intuitive. The 4-part, 16-chapter structure maps directly to the textbook requirements.

**Alternatives considered**:
- Flat structure: Would be harder to navigate for a textbook
- Topic-based structure: Less aligned with the specified 4-part organization

## Decision: Code Example Integration
**Rationale**: Python and ROS code examples will be integrated directly into the markdown files using code blocks with appropriate syntax highlighting. This ensures examples are part of the content and can be easily maintained alongside the text.

**Alternatives considered**:
- External files: Would make content harder to maintain
- Interactive playground: Too complex for Phase 1

## Decision: Navigation Structure
**Rationale**: Using Docusaurus sidebar navigation with a clear hierarchy reflecting the 4 parts and 16 chapters ensures users can easily navigate through the textbook content in a logical progression.

**Alternatives considered**:
- Single-page layout: Would make a 16-chapter textbook unwieldy
- Card-based navigation: Less suitable for structured textbook content

## Decision: Exercise Integration
**Rationale**: Exercises will be included at the end of each chapter as markdown sections, making them part of the chapter content but clearly demarcated. This follows educational best practices while maintaining the markdown format.

**Alternatives considered**:
- Separate exercise files: Would fragment the learning experience
- Interactive exercises: Too complex for Phase 1

## Technical Unknowns Resolved

### Docusaurus Version
- **Decision**: Use Docusaurus v3 (latest stable version)
- **Rationale**: Provides the most up-to-date features and security updates

### Content Authoring Process
- **Decision**: Use AI-assisted content generation with manual review
- **Rationale**: Aligns with the AI-first content creation principle while ensuring technical accuracy

### Diagram Integration
- **Decision**: Use static images with markdown image syntax
- **Rationale**: Simple to implement and maintain, compatible with GitHub Pages