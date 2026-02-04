/**
 * Remove class prefixes from right sidebar TOC entries
 * Changes "ArizeClient.datasets" to just "datasets" for cleaner display
 */
document.addEventListener('DOMContentLoaded', function() {
    // Find all TOC entries in the right sidebar
    const tocEntries = document.querySelectorAll('.bd-toc .toc-h3 code');

    tocEntries.forEach(function(entry) {
        const text = entry.textContent;
        // If the text contains a dot, keep only the part after the last dot
        if (text.includes('.')) {
            const parts = text.split('.');
            const shortName = parts[parts.length - 1];  // Get the last part
            entry.textContent = shortName;
        }
    });
});
